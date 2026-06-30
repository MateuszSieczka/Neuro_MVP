"""MuJoCo-MJX planar reaching arm — the canonical embodied plant.

A 2-link planar arm (shoulder + elbow), the minimal kinematic substrate for
target reaching (Mussa-Ivaldi & Giszter 1992; Shadmehr & Mussa-Ivaldi 1994).
Position-servo actuators abstract the α-motor-neuron pool (Lemon 2008), so
the brain's ``joint_command`` is a *desired joint angle* — tanh-bounded in
``[−1, 1]`` and rescaled to the anatomical range.

This is a pure :class:`~embodiment.body_interface.BodyInterface`: continuous
command in, a named sensory vector out (proprioception + absolute tip-position
population codes).  The brain cannot tell a "tip bump" from any other cortical
belief — the interface stays body-agnostic.

Import posture
--------------
The module imports without MuJoCo; only :meth:`MjxArmBody.create` triggers
the (lazy) import.  Windows dev boxes import the package and run the
substrate tests; MJX actually runs in the Colab notebook.  The physics step
(``mjx.step``) is JAX-native and composes with ``eqx.filter_jit`` and
``jax.lax.scan`` — the run loop drives many cycles under one compilation.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx

from core.backend import DTYPE, Array, PRNGKey
from sensory import (
    DEFAULT_VELOCITY_RANGE_FACTOR,
    ProprioceptionParams, init_proprioception_params,
    proprio_encode, monotonic_population_encode,
)

from .body_interface import (
    BodyInterface, SensorySample, SensoryLayout,
)


# Sensory segment names — goals address channels by these, never by index.
SEG_PROPRIOCEPTION = "proprioception"
SEG_TIP_X = "tip_x"
SEG_TIP_Y = "tip_y"
TIP_SEGMENTS = (SEG_TIP_X, SEG_TIP_Y)


# A compact 2-link planar reacher.  Geometry is in metres (forearm +
# upper-arm, 0.25 m each); the arm lies in the XY plane (gravity along −Z) so
# reaching does not fight gravity and babbling coverage is 2-D.  The numeric
# literals here are a physical-model specification (a MuJoCo XML), not control
# logic; the one brain-facing coupling — actuator ``ctrlrange`` ==
# ``ArmConfig.joint_range`` — is asserted in ``create``.
_ARM_XML = """<mujoco model="planar_reacher">
  <compiler angle="radian" autolimits="true"/>
  <!--
    MJX/T4 performance block: ``implicitfast`` integrator (stable at 10 ms,
    cheap for stiff position-servo + damping), ``CG`` solver with few
    iterations (no contacts to resolve), and the whole contact/equality
    pipeline disabled (the two capsules never self-collide; floor + target
    are contype=0) so MJX does not trace an empty collision kernel per step.
  -->
  <option timestep="0.01" gravity="0 0 -9.81"
          integrator="implicitfast"
          solver="CG" iterations="2" ls_iterations="4">
    <flag contact="disable" equality="disable" frictionloss="disable"/>
  </option>
  <default>
    <joint damping="0.2" armature="0.01"/>
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

#: ``ctrlrange`` declared in ``_ARM_XML`` — must equal ``ArmConfig.joint_range``
#: (the command rescaling and the actuator range have to agree).
_XML_CTRLRANGE = 2.0


class ArmConfig(NamedTuple):
    """Static configuration of the MJX arm body."""

    n_joints: int
    motor_dim: int
    n_cells_per_joint: int        # population cells per joint, angle & velocity
    n_target_cells: int           # population cells per target-error axis
    frame_skip: int               # physics sub-steps per brain cycle
    max_steps: int                # episode length (cycles)
    include_target_in_sensory: bool
    joint_range: float            # symmetric ±range the command spans (rad)
    workspace_half: float         # spatial half-extent of the 2-D reach area (m)
    reach_annulus: tuple[float, float]  # (min, max) target radius (m)
    target_z: float               # target / arm plane height (m)


def default_arm_config(
    *,
    include_target: bool = True,
    n_cells_per_joint: int = 16,
    n_target_cells: int = 16,
    frame_skip: int = 3,
    max_steps: int = 500,
) -> ArmConfig:
    """Default 2-DOF planar reacher configuration.

    ``joint_range`` ≈ anatomical shoulder/elbow flexion; the reach annulus
    keeps targets strictly inside the 0.5 m two-link reach.
    """
    return ArmConfig(
        n_joints=2,
        motor_dim=2,
        n_cells_per_joint=n_cells_per_joint,
        n_target_cells=n_target_cells,
        frame_skip=frame_skip,
        max_steps=max_steps,
        include_target_in_sensory=include_target,
        joint_range=2.0,
        workspace_half=0.5,
        reach_annulus=(0.15, 0.45),
        target_z=0.05,
    )


def _arm_sensory_layout(cfg: ArmConfig) -> SensoryLayout:
    """Named partition: proprioception, then per-axis absolute tip-position codes.

    The tip channels are the *controllable, learnable* coordinate frame the
    reach goal lives in: ``tip = FK(joint angles)`` is a pure function of the
    motor command, so the ``motor→cerebellum→sensory`` forward model can
    predict it (and active inference can invert it) — unlike an absolute
    target, which is exogenous.  The tip uses a **monotonic** population code
    (:func:`sensory.monotonic_population_encode`) so the clamped reach goal is
    invertible everywhere; proprioception stays a Gaussian code (it is an
    afferent read-out, not an inversion target).  The goal ("tip at target")
    is a target-specific clamp on these channels (:meth:`MjxArmBody.reach_goal`).
    """
    proprio_size = cfg.n_joints * 2 * cfg.n_cells_per_joint
    named: list[tuple[str, int]] = [(SEG_PROPRIOCEPTION, proprio_size)]
    if cfg.include_target_in_sensory:
        named.append((SEG_TIP_X, cfg.n_target_cells))
        named.append((SEG_TIP_Y, cfg.n_target_cells))
    return SensoryLayout.from_sizes(tuple(named))


class MjxArmBody(eqx.Module, BodyInterface):
    """JAX-native MuJoCo planar arm conforming to ``BodyInterface``.

    A value object: ``act`` / ``reset`` return a *new* body with updated
    ``mjx_data``; the static ``mj_model`` (CPU, rendering only) and the
    array-pytree ``mjx_model`` are carried through unchanged.
    """

    # Dynamic pytree leaves (traced under JIT):
    mjx_data: Any                           # mujoco.mjx.Data — changes every step
    target_xy: Array                        # (2,) float32, arm-plane target
    step_idx: Array                         # scalar int32
    # JAX-array pytree leaves — constant across a run but NOT static (marking
    # an mjx.Model static makes equinox hash its array contents every call →
    # cache-miss + full recompile per step; left as leaves XLA folds once).
    mjx_model: Any                          # mujoco.mjx.Model (JAX pytree)
    proprio: ProprioceptionParams           # Gaussian-tuning params (JAX arrays)
    # Pure-Python / C-object static leaves — no JAX arrays here:
    mj_model: Any = eqx.field(static=True)  # CPU mujoco.MjModel (rendering only)
    cfg: ArmConfig = eqx.field(static=True)
    sensory_layout: SensoryLayout = eqx.field(static=True)
    sensory_size: int = eqx.field(static=True)
    motor_dim: int = eqx.field(static=True)
    mocap_target_body_id: int = eqx.field(static=True)
    tip_site_id: int = eqx.field(static=True)

    # ------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        *,
        cfg: ArmConfig | None = None,
        xml: str | None = None,
    ) -> "MjxArmBody":
        """Build a fresh arm body.  Imports MuJoCo lazily."""
        try:                                # lazy import — not at module load
            import mujoco
            from mujoco import mjx
        except ImportError as e:            # pragma: no cover
            raise RuntimeError(
                "MjxArmBody.create requires `mujoco` + `mujoco-mjx`. "
                "Install with `pip install mujoco mujoco-mjx` (the Colab "
                "notebook does this automatically)."
            ) from e

        cfg = cfg or default_arm_config()
        if xml is None and cfg.joint_range != _XML_CTRLRANGE:
            raise ValueError(
                f"cfg.joint_range ({cfg.joint_range}) must match the arm XML "
                f"actuator ctrlrange (±{_XML_CTRLRANGE}); pass a matching "
                f"`xml` to use a different range."
            )

        mj_model = mujoco.MjModel.from_xml_string(xml or _ARM_XML)
        mjx_model = mjx.put_model(mj_model)
        mjx_data = mjx.make_data(mjx_model)
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
            velocity_range=(
                -cfg.joint_range * DEFAULT_VELOCITY_RANGE_FACTOR,
                cfg.joint_range * DEFAULT_VELOCITY_RANGE_FACTOR,
            ),
        )

        layout = _arm_sensory_layout(cfg)
        key, sub = jax.random.split(key)
        target = _sample_target(sub, cfg)

        body = cls(
            mjx_data=mjx_data,
            target_xy=target,
            step_idx=jnp.asarray(0, jnp.int32),
            mjx_model=mjx_model,
            proprio=proprio,
            mj_model=mj_model,
            cfg=cfg,
            sensory_layout=layout,
            sensory_size=layout.total,
            motor_dim=cfg.motor_dim,
            mocap_target_body_id=mocap_body_id,
            tip_site_id=tip_site_id,
        )
        return body._set_target(target)

    # ------------------------------------------------------------ #
    # BodyInterface API
    # ------------------------------------------------------------ #

    @property
    def layout(self) -> SensoryLayout:
        return self.sensory_layout

    def reset(self, key: PRNGKey) -> tuple["MjxArmBody", SensorySample]:
        from mujoco import mjx
        k_t, _ = jax.random.split(key)
        target = _sample_target(k_t, self.cfg)
        data0 = mjx.make_data(self.mjx_model)   # zeros joint state + velocities
        body = eqx.tree_at(
            lambda b: (b.mjx_data, b.step_idx, b.target_xy),
            self,
            (data0, jnp.asarray(0, jnp.int32), target),
        )._set_target(target)
        sensory, _, tip = body._observe(reward=False)
        return body, SensorySample(
            sensory=sensory,
            reward=jnp.asarray(0.0, DTYPE),
            done=jnp.asarray(0.0, DTYPE),
            info={"tip": tip, "target": target},
        )

    def act(
        self, key: PRNGKey, joint_command: Array,
    ) -> tuple["MjxArmBody", SensorySample]:
        """Step physics under a desired-joint-angle command.

        ``joint_command`` is ``(motor_dim,)`` tanh-bounded in ``[−1, 1]``;
        rescaled to the actuator range ±``joint_range`` (radians) and held
        for ``frame_skip`` physics sub-steps.
        """
        from mujoco import mjx
        del key                              # arm dynamics are deterministic
        jc = jnp.clip(
            jnp.asarray(joint_command, DTYPE), -1.0, 1.0,
        )[: self.cfg.motor_dim]
        ctrl = jc * jnp.asarray(self.cfg.joint_range, DTYPE)
        data = self.mjx_data.replace(ctrl=ctrl)
        mjx_model = self.mjx_model

        # ``fori_loop`` keeps a single compiled copy of ``mjx.step`` (a Python
        # for-loop unrolls into frame_skip copies and inflates compile time).
        def _phys_step(_i, d):
            return mjx.step(mjx_model, d)

        data = jax.lax.fori_loop(
            0, jnp.asarray(int(self.cfg.frame_skip), jnp.int32),
            _phys_step, data,
        )

        new_step = self.step_idx + jnp.asarray(1, jnp.int32)
        body = eqx.tree_at(
            lambda b: (b.mjx_data, b.step_idx), self, (data, new_step),
        )
        sensory, reward, tip = body._observe(reward=True)
        done = (new_step >= self.cfg.max_steps).astype(DTYPE)
        return body, SensorySample(
            sensory=sensory,
            reward=reward,
            done=done,
            info={"tip": tip, "target": body.target_xy, "command": jc},
        )

    # ------------------------------------------------------------ #
    # Goal specification
    # ------------------------------------------------------------ #

    def reach_goal(self) -> tuple[Array, Array]:
        """Partial sensory preference for "tip on target" + its pin mask.

        Returns ``(preference, mask)`` for :func:`core.pc_brain.pc_brain_act`:
        the absolute tip-position channels are pinned to the **monotonic**
        population code of the **observed target** position; proprioception is
        left free for the brain to infer.  Target-**specific** — the goal
        carries the target in the tip coordinate frame, so relaxing the
        (flat-prior) motor node inverts the ``motor→cerebellum→tip`` forward
        model to a target-dependent command.

        This is the active-inference reach of Adams, Shipp & Friston (2013):
        set the desired (proprioceptive/tip) outcome, infer the command that
        realises it.  The tip uses the *monotonic* code (not a Gaussian bump)
        precisely because active inference inverts it: a bump goal makes the
        free-energy (L2) gradient vanish wherever the predicted tip is off the
        target bump, so the command is never driven; the monotonic code keeps
        the gradient non-zero across the whole workspace
        (:func:`sensory.monotonic_population_encode`).
        """
        if not self.cfg.include_target_in_sensory:
            raise ValueError(
                "reach_goal requires the tip-position sensory channels; "
                "build the body with include_target=True"
            )
        half = self.cfg.workspace_half
        tgt_x_code = monotonic_population_encode(
            self.target_xy[0], self.cfg.n_target_cells, x_min=-half, x_max=half,
        )
        tgt_y_code = monotonic_population_encode(
            self.target_xy[1], self.cfg.n_target_cells, x_min=-half, x_max=half,
        )
        preference = jnp.zeros(self.sensory_layout.total, DTYPE)
        seg_x = self.sensory_layout.segment(SEG_TIP_X)
        seg_y = self.sensory_layout.segment(SEG_TIP_Y)
        preference = preference.at[seg_x.start:seg_x.stop].set(tgt_x_code)
        preference = preference.at[seg_y.start:seg_y.stop].set(tgt_y_code)
        mask = self.sensory_layout.mask(TIP_SEGMENTS)
        return preference, mask

    # ------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------ #

    def _set_target(self, target_xy: Array) -> "MjxArmBody":
        """Write the sampled target into the single mocap slot of mjx_data."""
        t3 = jnp.concatenate(
            [target_xy, jnp.asarray([self.cfg.target_z], DTYPE)],
        )
        mocap_pos = self.mjx_data.mocap_pos.at[0].set(t3)
        data = self.mjx_data.replace(mocap_pos=mocap_pos)
        return eqx.tree_at(lambda b: b.mjx_data, self, data)

    def _observe(self, *, reward: bool) -> tuple[Array, Array, Array]:
        """Build the sensory vector + (optional) reward + tip position."""
        qpos = self.mjx_data.qpos[: self.cfg.n_joints].astype(DTYPE)
        qvel = self.mjx_data.qvel[: self.cfg.n_joints].astype(DTYPE)
        proprio = proprio_encode(self.proprio, qpos, qvel)
        tip_xy = self.mjx_data.site_xpos[self.tip_site_id, :2].astype(DTYPE)

        if self.cfg.include_target_in_sensory:
            half = self.cfg.workspace_half
            # Absolute tip position (a pure function of the joint angles, hence
            # of the motor command) — the learnable frame the reach goal lives
            # in.  Tip ∈ [−(L1+L2), L1+L2] = [−half, half] for this arm.  The
            # *monotonic* code (not a Gaussian bump) so the reach goal clamped
            # on these channels stays invertible (non-vanishing free-energy
            # gradient) across the whole workspace.
            tx_code = monotonic_population_encode(
                tip_xy[0], self.cfg.n_target_cells, x_min=-half, x_max=half,
            )
            ty_code = monotonic_population_encode(
                tip_xy[1], self.cfg.n_target_cells, x_min=-half, x_max=half,
            )
            sensory = jnp.concatenate([proprio, tx_code, ty_code])
        else:
            sensory = proprio

        r = (-jnp.linalg.norm(self.target_xy - tip_xy)).astype(DTYPE) if reward \
            else jnp.asarray(0.0, DTYPE)
        return sensory.astype(DTYPE), r, tip_xy

    # Convenience access for drivers / tests.
    def tip_xy(self) -> Array:
        return self.mjx_data.site_xpos[self.tip_site_id, :2].astype(DTYPE)

    def qpos(self) -> Array:
        return self.mjx_data.qpos[: self.cfg.n_joints].astype(DTYPE)

    def qvel(self) -> Array:
        return self.mjx_data.qvel[: self.cfg.n_joints].astype(DTYPE)


def _sample_target(key: PRNGKey, cfg: ArmConfig) -> Array:
    """Sample a random reachable target in the planar workspace annulus."""
    k_r, k_th = jax.random.split(key)
    lo, hi = cfg.reach_annulus
    r = jax.random.uniform(
        k_r, (), DTYPE, lo, min(hi, 0.9 * cfg.workspace_half),
    )
    th = jax.random.uniform(k_th, (), DTYPE, -jnp.pi, jnp.pi)
    return jnp.stack([r * jnp.cos(th), r * jnp.sin(th)])
