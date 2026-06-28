"""Phase 6B — MuJoCo-MJX planar reaching arm body.

This is the Phase 6B drop-in that replaces the discrete gridworld body
with a real continuous-torque (position-servo) physics sim.  The file
is designed to be importable even when MuJoCo is not installed: the
top-level module has no hard MuJoCo dependency; only the
``MjxArmBody.create`` constructor triggers the import.  That way
Windows development machines can still import the package and run the
rest of the test-suite; the Colab notebook is where MJX actually runs.

Biological framing
------------------
A 2-link planar arm (shoulder + elbow) is the minimal kinematic
substrate for target reaching; this is the preparation used by
Mussa-Ivaldi & Giszter (1992) for spinal motor primitives, and by
Shadmehr & Mussa-Ivaldi (1994) for force-field adaptation.  We use
position-servo actuators (abstracting the α-motor-neuron pool below
the spinal cord; Lemon 2008) so the brain's ``joint_command`` is a
*desired joint angle*, not a raw torque.  ``joint_command`` in
``[-1, 1]`` is rescaled to each joint's kinematic range.

Sensory coding
--------------
Proprioception is encoded via the Phase 6A Gaussian population code
(``sensory.proprioception.proprio_encode``).  Optional end-effector-
to-target delta is appended as a 2-D Gaussian population code so the
brain has a target-conditioned sensory feature *without* supplying
privileged (x, y) coordinates in raw form.  This keeps the interface
body-agnostic: the brain can't tell the difference between "target
delta encoded as population bump" and "visual cortex belief".

JIT posture
-----------
The physics step (``mujoco.mjx.step``) is JAX-native and composes with
our existing ``eqx.filter_jit`` machinery.  We wrap a fresh ``jit``
around the physics step only (not around the brain), so compile cost
is bounded and the body integrates naturally into ``run_loop``.
``mujoco.Renderer`` is CPU-only and is called at user-controlled
cadence from the notebook driver; it is **not** part of any JIT path.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx

from core.backend import DTYPE, Array, PRNGKey
from sensory.proprioception import (
    ProprioceptionParams, init_proprioception_params,
    proprio_encode, proprio_output_dim,
)

from .body_interface import BodyInterface, SensorySample, gauss_pop_encode


# A compact 2-link planar reacher.  Geometry roughly matches a forearm
# + upper-arm in metres; actuator ranges cover anatomical joint limits.
# We keep the arm in the XY plane (gravity along -Z, arm horizontal) so
# reaching does not fight gravity and babbling coverage is 2-D.
_ARM_XML = """<mujoco model="planar_reacher">
  <compiler angle="radian" autolimits="true"/>
  <!--
    MJX/T4 performance-critical option block.

    * ``integrator="implicitfast"`` — MJX-recommended integrator for
      stiff position-servo actuators + linear damping; stable at 10 ms
      and ~5–20× cheaper per ``mjx.step`` than the CPU default
      (semi-implicit Euler).  Matches the guidance in the
      mujoco-mjx/benchmarks docs.
    * ``solver="CG"`` with ``iterations="2"`` / ``ls_iterations="4"``
      replaces the CPU default Newton-CG (100 iters, 1e-8 tol) which
      compiles a very large per-step graph on MJX.  The 2-link arm
      has no contacts and only damping/servo torques → 2 iterations
      converge.
    * ``<flag contact="disable" equality="disable"
            frictionloss="disable"/>`` kills the entire contact /
      equality pipeline.  Floor + target are already ``contype=0``
      and the two arm capsules are not intended to self-collide, so
      this is physically equivalent — it just stops MJX from tracing
      an empty collision kernel on every step.
  -->
  <option timestep="0.01" gravity="0 0 -9.81"
          integrator="implicitfast"
          solver="CG" iterations="2" ls_iterations="4">
    <flag contact="disable" equality="disable" frictionloss="disable"/>
  </option>
  <default>
    <joint damping="0.2" armature="0.01"/>
    <!--
      ``ctrlrange`` matches ``ArmConfig.joint_range=±2.0 rad`` so the
      brain's tanh output (rescaled by ``joint_range`` inside
      ``act_continuous``) can command the full anatomical joint
      range.  An earlier ``ctrlrange="-1 1"`` silently truncated
      every command to ±57°, so the elbow could never fold enough to
      reach negative-x or deep-y targets — tip bbox collapsed to a
      ~2 cm band during babbling.
    -->
    <position ctrlrange="-2.0 2.0" kp="20" kv="4"/>
    <geom rgba="0.7 0.7 0.9 1"/>
  </default>
  <worldbody>
    <light pos="0 0 2"/>
    <camera name="topdown" pos="0 0 1.5" xyaxes="1 0 0 0 1 0"/>
    <geom name="floor" type="plane" size="1 1 0.02" rgba="0.95 0.95 0.95 1"/>
    <body name="shoulder" pos="0 0 0.05">
      <joint name="j0" type="hinge" axis="0 0 1" range="-2.3 2.3"/>
      <geom name="upper" type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"/>
      <body name="elbow" pos="0.25 0 0">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2.3 2.3"/>
        <geom name="fore" type="capsule" fromto="0 0 0 0.25 0 0" size="0.025"/>
        <site name="tip" pos="0.25 0 0" size="0.02" rgba="1 0.2 0.2 1"/>
      </body>
    </body>
    <body name="target" pos="0.3 0.2 0.05" mocap="true">
      <geom name="target" type="sphere" size="0.03" rgba="0.2 0.9 0.2 1"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a0" joint="j0"/>
    <position name="a1" joint="j1"/>
  </actuator>
</mujoco>
"""


class ArmConfig(NamedTuple):
    """Static configuration of the MJX arm body."""
    n_joints: int
    motor_dim: int
    n_cells_per_joint: int
    n_target_cells: int          # per axis, Gaussian code of Δtarget
    frame_skip: int              # physics sub-steps per brain cycle
    max_steps: int
    include_target_in_sensory: bool
    joint_range: float           # symmetric ±range the command spans
    workspace_half: float        # spatial half-extent of the 2-D reach area


def default_arm_config(
    *,
    include_target: bool = True,
    n_cells_per_joint: int = 16,
    n_target_cells: int = 16,
    frame_skip: int = 3,
    max_steps: int = 500,
) -> ArmConfig:
    """Default 2-DOF planar reacher configuration."""
    return ArmConfig(
        n_joints=2,
        motor_dim=2,
        n_cells_per_joint=n_cells_per_joint,
        n_target_cells=n_target_cells,
        frame_skip=frame_skip,
        max_steps=max_steps,
        include_target_in_sensory=include_target,
        joint_range=2.0,          # ~ anatomical shoulder/elbow flexion
        workspace_half=0.5,
    )


def _sensory_size_for(cfg: ArmConfig) -> int:
    p_sz = cfg.n_joints * 2 * cfg.n_cells_per_joint   # angles + velocities
    t_sz = (2 * cfg.n_target_cells) if cfg.include_target_in_sensory else 0
    return int(p_sz + t_sz)


# --------------------------------------------------------------------
# Lazy MJX handles — stored inside the body as *opaque* pytree leaves
# via ``eqx.field(static=True)`` so they survive JIT but are not traced.
# --------------------------------------------------------------------


class MjxArmBody(eqx.Module, BodyInterface):
    """JAX-native MuJoCo planar arm, conforming to ``BodyInterface``.

    The body is a value object: ``act`` / ``act_continuous`` / ``reset``
    all return a *new* body instance with updated ``mjx_data``.  The
    static ``mjx_model`` and ``mj_model`` handles are carried through
    every replacement unchanged.
    """

    # Dynamic pytree leaves (traced under JIT):
    mjx_data: Any                           # mujoco.mjx.Data — changes every step
    target_xy: Array                        # (2,) float32, arm plane target
    step_idx: Array                         # scalar int32
    # JAX-array pytree leaves — constant across a run but NOT static:
    # marking eqx.Module / mjx.Model as static=True causes equinox to
    # hash the full array contents as a Python object on every call,
    # producing a cache-miss and full XLA recompile each step.
    # Leaving them as regular leaves lets XLA constant-fold them once.
    mjx_model: Any                          # mujoco.mjx.Model (JAX pytree)
    proprio: ProprioceptionParams           # Gaussian-tuning params (JAX arrays)
    # Pure-Python / C-object static leaves — no JAX arrays here:
    mj_model: Any = eqx.field(static=True)  # CPU mujoco.MjModel (rendering only)
    cfg: ArmConfig = eqx.field(static=True)
    sensory_size: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)
    mocap_target_body_id: int = eqx.field(static=True)
    tip_site_id: int = eqx.field(static=True)

    # ------------------------------------------------------------ #
    # Construction                                                 #
    # ------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        *,
        cfg: ArmConfig | None = None,
        xml: str | None = None,
        n_actions: int | None = None,
    ) -> "MjxArmBody":
        """Build a fresh arm body.  Imports MuJoCo lazily.

        ``n_actions`` is the discrete body-action count the BG
        body-actor uses (regression-safe discretisation path).  When
        ``bypass_m1=False`` the continuous ``joint_command`` is used
        instead and ``n_actions`` is consulted only by the discrete
        argmax adapter (Phase 6A).
        """
        try:                                # lazy import — not at module load
            import mujoco
            from mujoco import mjx
        except ImportError as e:            # pragma: no cover
            raise RuntimeError(
                "MjxArmBody.create requires `mujoco` + `mujoco-mjx`. "
                "Install with `pip install mujoco mujoco-mjx` (Phase 6B "
                "Colab notebook does this automatically)."
            ) from e

        cfg = cfg or default_arm_config()
        mj_model = mujoco.MjModel.from_xml_string(xml or _ARM_XML)
        mjx_model = mjx.put_model(mj_model)
        mjx_data = mjx.make_data(mjx_model)
        # Identify mocap target body + end-effector site once.
        mocap_body_id = int(
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "target")
        )
        tip_site_id = int(
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        )

        proprio = init_proprioception_params(
            n_joints=cfg.n_joints,
            n_cells_per_joint=cfg.n_cells_per_joint,
            angle_range=(-cfg.joint_range, cfg.joint_range),
            velocity_range=(-cfg.joint_range * 4.0, cfg.joint_range * 4.0),
        )

        sensory_size = _sensory_size_for(cfg)
        # One discrete action per sign-split motor DoF, minimum 4 so the
        # BG body-actor has meaningful categorical branches even when
        # only the continuous M1 head is driving (regression path).
        n_act = int(n_actions or max(4, 2 * cfg.motor_dim))

        key, sub = jax.random.split(key)
        target = _sample_target(sub, cfg.workspace_half)

        body = cls(
            mjx_data=mjx_data,
            target_xy=target,
            step_idx=jnp.asarray(0, jnp.int32),
            mjx_model=mjx_model,
            mj_model=mj_model,
            proprio=proprio,
            cfg=cfg,
            sensory_size=sensory_size,
            n_actions=n_act,
            mocap_target_body_id=mocap_body_id,
            tip_site_id=tip_site_id,
        )
        return body._set_target(target)

    # ------------------------------------------------------------ #
    # BodyInterface API                                            #
    # ------------------------------------------------------------ #

    def reset(self, key: PRNGKey) -> tuple["MjxArmBody", SensorySample]:
        from mujoco import mjx
        k_t, _ = jax.random.split(key)
        target = _sample_target(k_t, self.cfg.workspace_half)
        # Fresh data — zeros out joint state and velocities.
        data0 = mjx.make_data(self.mjx_model)
        body = eqx.tree_at(
            lambda b: (b.mjx_data, b.step_idx, b.target_xy),
            self,
            (data0, jnp.asarray(0, jnp.int32), target),
        )._set_target(target)
        sensory, reward, tip = body._observe(reward=False)
        return body, SensorySample(
            sensory=sensory,
            reward=jnp.asarray(0.0, DTYPE),
            done=jnp.asarray(0.0, DTYPE),
            info={"tip": tip, "target": target},
        )

    def act(
        self,
        key: PRNGKey,
        body_action: Array,
        saccade_action: Array,
    ) -> tuple["MjxArmBody", SensorySample]:
        """Discrete-action fallback path (Phase 6A regression compat).

        ``body_action`` is interpreted as a sign-split argmax: the
        lower half of ``[0, n_actions)`` picks a positive DoF, the
        upper half a negative one.  This makes ``bypass_m1=True``
        usable against MJX as well, at reduced expressiveness.
        """
        n_half = self.cfg.motor_dim
        a = jnp.asarray(body_action, jnp.int32)
        a = jnp.clip(a, 0, 2 * n_half - 1)
        sign = jnp.where(a < n_half, 1.0, -1.0)
        idx = jnp.where(a < n_half, a, a - n_half)
        one_hot = (jnp.arange(self.cfg.motor_dim) == idx).astype(DTYPE)
        jc = (sign * one_hot).astype(DTYPE)
        return self.act_continuous(key, jc)

    def act_continuous(
        self,
        key: PRNGKey,
        joint_command: Array,
    ) -> tuple["MjxArmBody", SensorySample]:
        """Phase 6B primary path: step physics with an M1 command.

        ``joint_command`` is ``(motor_dim,)`` tanh-bounded in ``[-1, 1]``
        and is interpreted as a *normalised desired joint angle*.  We
        rescale into actuator ctrlrange (symmetric ±joint_range) and
        advance the simulator by ``cfg.frame_skip`` physics steps.
        """
        from mujoco import mjx
        jc = jnp.clip(
            jnp.asarray(joint_command, DTYPE),
            -1.0, 1.0,
        )[: self.cfg.motor_dim]
        # ``position`` actuators treat ``ctrl`` as the desired joint
        # *setpoint in radians*.  The brain-side contract is that
        # ``joint_command`` is tanh-bounded in [-1, 1]; we rescale it
        # here to the anatomical range ±``cfg.joint_range`` before
        # writing ``data.ctrl``.  The XML ctrlrange is widened to
        # match (see ``_ARM_XML``).
        ctrl = jc * jnp.asarray(self.cfg.joint_range, DTYPE)
        data = self.mjx_data
        data = data.replace(ctrl=ctrl)
        # Advance ``frame_skip`` physics steps using ``jax.lax.fori_loop``.
        # Earlier versions used a Python ``for`` loop which, under an
        # outer ``jit``/``scan``, unrolled into ``frame_skip`` full
        # copies of ``mjx.step`` and inflated XLA compile time by
        # ``frame_skip×``.  ``fori_loop`` keeps a single compiled copy.
        mjx_model = self.mjx_model

        def _phys_step(_i, d):
            return mjx.step(mjx_model, d)

        data = jax.lax.fori_loop(
            0, jnp.asarray(int(self.cfg.frame_skip), jnp.int32),
            _phys_step, data,
        )

        new_step = self.step_idx + jnp.asarray(1, jnp.int32)
        body = eqx.tree_at(
            lambda b: (b.mjx_data, b.step_idx),
            self, (data, new_step),
        )
        sensory, reward, tip = body._observe(reward=True)
        done = (new_step >= self.cfg.max_steps).astype(DTYPE)
        return body, SensorySample(
            sensory=sensory,
            reward=reward,
            done=done,
            info={"tip": tip, "target": body.target_xy, "ctrl": jc},
        )

    # ------------------------------------------------------------ #
    # Helpers                                                      #
    # ------------------------------------------------------------ #

    def _set_target(self, target_xy: Array) -> "MjxArmBody":
        """Write the sampled target into the mocap slot of mjx_data."""
        t3 = jnp.concatenate(
            [target_xy, jnp.asarray([0.05], DTYPE)]
        )[None, :]                                             # (1, 3)
        # ``mocap_pos`` is (n_mocap, 3); we have a single mocap body.
        mocap_pos = self.mjx_data.mocap_pos.at[0].set(t3[0])
        data = self.mjx_data.replace(mocap_pos=mocap_pos)
        return eqx.tree_at(lambda b: b.mjx_data, self, data)

    def _observe(self, *, reward: bool) -> tuple[Array, Array, Array]:
        """Build sensory vector + (optional) reward + tip position."""
        qpos = self.mjx_data.qpos[: self.cfg.n_joints].astype(DTYPE)
        qvel = self.mjx_data.qvel[: self.cfg.n_joints].astype(DTYPE)
        proprio = proprio_encode(self.proprio, qpos, qvel)
        tip_xy = self.mjx_data.site_xpos[self.tip_site_id, :2].astype(DTYPE)

        if self.cfg.include_target_in_sensory:
            dxy = jnp.clip(
                self.target_xy - tip_xy,
                -2.0 * self.cfg.workspace_half,
                2.0 * self.cfg.workspace_half,
            )
            n = self.cfg.n_target_cells
            half = self.cfg.workspace_half
            dx_code = gauss_pop_encode(
                dxy[0], n, x_min=-half, x_max=half,
            )
            dy_code = gauss_pop_encode(
                dxy[1], n, x_min=-half, x_max=half,
            )
            sensory = jnp.concatenate([proprio, dx_code, dy_code])
        else:
            sensory = proprio

        if reward:
            dist = jnp.linalg.norm(self.target_xy - tip_xy)
            r = (-dist).astype(DTYPE)
        else:
            r = jnp.asarray(0.0, DTYPE)
        return sensory.astype(DTYPE), r, tip_xy

    # Convenience access for tests / drivers.
    def tip_xy(self) -> Array:
        return self.mjx_data.site_xpos[self.tip_site_id, :2].astype(DTYPE)

    def qpos(self) -> Array:
        return self.mjx_data.qpos[: self.cfg.n_joints].astype(DTYPE)

    def qvel(self) -> Array:
        return self.mjx_data.qvel[: self.cfg.n_joints].astype(DTYPE)


def _sample_target(key: PRNGKey, workspace_half: float) -> Array:
    """Sample a random reachable target in the planar workspace annulus."""
    k_r, k_th = jax.random.split(key)
    # Annulus [0.15, 0.45] metres in the planar workspace keeps the
    # target strictly inside reach (forearm+upper-arm each 0.25 m).
    r = jax.random.uniform(
        k_r, (), DTYPE, 0.15, min(0.45, 0.9 * workspace_half)
    )
    th = jax.random.uniform(k_th, (), DTYPE, -jnp.pi, jnp.pi)
    return jnp.stack([r * jnp.cos(th), r * jnp.sin(th)])
