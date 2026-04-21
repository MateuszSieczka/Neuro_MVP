"""Phase 6B common test scaffolding — skip if mujoco/mjx missing."""
import pytest

# All tests in this folder that target Phase 6B gate on the availability
# of MuJoCo + MJX.  On Colab T4 these are pip-installable; on Windows
# they are not easily available.
mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")
mjx = pytest.importorskip("mujoco.mjx", reason="mujoco-mjx not installed")
