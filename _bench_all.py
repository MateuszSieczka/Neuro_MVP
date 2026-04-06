"""Combined benchmark: CartPole + MountainCar — tests generality of architecture."""
import subprocess, sys

print("=" * 70)
print("  BENCHMARK 1: CartPole-v1  (dense reward, 200 episodes)")
print("=" * 70)
subprocess.run([sys.executable, "_sweep.py"], check=True)

print("\n" + "=" * 70)
print("  BENCHMARK 2: MountainCar-v0  (sparse reward, 500 episodes)")
print("=" * 70)
subprocess.run([sys.executable, "_sweep_mc.py"], check=True)
