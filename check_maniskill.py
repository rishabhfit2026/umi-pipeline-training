"""
Run this first to check your ManiSkill setup:
    source /home/rishabh/Downloads/umi-pipeline-training/umi_env/bin/activate
    python check_maniskill.py
"""
# Check version and available envs
try:
    import mani_skill
    print("ManiSkill version:", mani_skill.__version__)
except: pass

try:
    import mani_skill2
    print("ManiSkill2 version:", mani_skill2.__version__)
except: pass

# Check what's available
try:
    import gymnasium as gym
    import mani_skill.envs
    envs = [e for e in gym.envs.registry.keys() if "maniskill" in e.lower() or "pick" in e.lower()]
    print("Available envs:", envs[:10])
except Exception as e:
    print("gym error:", e)

# Check SAPIEN (physics engine under ManiSkill)
try:
    import sapien
    print("SAPIEN version:", sapien.__version__)
except Exception as e:
    print("SAPIEN:", e)

# Check render
try:
    import mani_skill.envs.tasks
    print("Tasks available")
except Exception as e:
    print("Tasks:", e)

print("\nDone.")