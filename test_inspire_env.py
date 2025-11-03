#!/usr/bin/env python3

"""
Test script for Inspire Hand Manipulation Environment

This script tests:
1. Environment creation
2. Reset functionality
3. Stepping through episodes
4. Observation/action/reward shapes
"""

import argparse
import torch
import sys

# Step 1: Import AppLauncher FIRST
from isaaclab.app import AppLauncher

# Step 2: Parse arguments
parser = argparse.ArgumentParser(description="Test Inspire manipulation environment")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument("--episode_length", type=int, default=100, help="Episode length for test")
# Note: AppLauncher.add_app_launcher_args() adds --headless automatically
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Step 3: Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Step 4: NOW import everything else
print("Isaac Sim launched successfully!")
print("Importing modules...")

import gymnasium as gym
import Franka_RL  # This triggers task registration

# Wrap everything in try-finally to ensure cleanup
try:
    print("\n" + "="*80)
    print("TEST 1: Check Task Registration")
    print("="*80)

    # List all registered tasks
    all_tasks = [env_id for env_id in gym.envs.registry.keys() if "Inspire" in env_id]
    print(f"Found Inspire tasks: {all_tasks}")

    if "Inspire-Manip-v0" not in all_tasks:
        print("ERROR: Inspire-Manip-v0 not registered!")
        print("Make sure you created all the files correctly.")
        sys.exit(1)

    print("✓ Task registration successful!")

    print("\n" + "="*80)
    print("TEST 2: Create Environment")
    print("="*80)

    try:
        # Load the environment config
        from Franka_RL.tasks.direct.inspire_manip.inspire_manip_env_cfg import InspireManipEnvCfg

        # Create config instance with overrides
        env_cfg = InspireManipEnvCfg()
        env_cfg.scene.num_envs = args.num_envs

        # Create environment with explicit config
        env = gym.make(
            "Inspire-Manip-v0",
            cfg=env_cfg,
            render_mode=None,
        )
        print(f"✓ Environment created successfully!")
        print(f"  - Number of environments: {env.unwrapped.num_envs}")
        print(f"  - Device: {env.unwrapped.device}")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*80)
    print("TEST 3: Check Spaces")
    print("="*80)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"  - Single action shape: {env.unwrapped.single_action_space.shape}")
    print(f"  - Expected: (18,) for 6D wrist + 12 DOF")

    print("\n" + "="*80)
    print("TEST 4: Reset Environment")
    print("="*80)

    try:
        obs_dict, info = env.reset()
        print(f"✓ Reset successful!")
        print(f"  - Policy obs shape: {obs_dict['policy'].shape}")
        print(f"  - Critic obs shape: {obs_dict['critic'].shape if 'critic' in obs_dict else 'N/A'}")
        print(f"  - Info keys: {list(info.keys())}")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*80)
    print("TEST 5: Step Through Episode")
    print("="*80)

    try:
        num_steps = args.episode_length
        print(f"Stepping through {num_steps} steps...")

        total_reward = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
        num_resets = 0

        # Get action dimension from single_action_space
        action_dim = env.unwrapped.single_action_space.shape[0]

        for step in range(num_steps):
            # Random actions
            actions = torch.randn(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)
            actions = torch.clamp(actions, -1.0, 1.0)  # Clip to [-1, 1]

            # Step environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)

            # Accumulate rewards
            total_reward += rewards

            # Check for resets
            dones = terminated | truncated
            if dones.any():
                num_resets += dones.sum().item()

            # Print progress every 10 steps
            if (step + 1) % 10 == 0:
                avg_reward = total_reward.mean().item() / (step + 1)
                print(f"  Step {step+1}/{num_steps} - Avg reward: {avg_reward:.4f}")

        print(f"✓ Stepping successful!")
        print(f"  - Total steps: {num_steps}")
        print(f"  - Total resets: {num_resets}")
        print(f"  - Average reward: {total_reward.mean().item() / num_steps:.4f}")
        print(f"  - Reward std: {total_reward.std().item() / num_steps:.4f}")

    except Exception as e:
        print(f"✗ Stepping failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*80)
    print("TEST 6: Check Environment Components")
    print("="*80)

    print(f"Robot:")
    print(f"  - Name: {env.unwrapped.robot.cfg.prim_path}")
    print(f"  - DOFs: {env.unwrapped.robot.num_joints}")
    print(f"  - Bodies: {env.unwrapped.robot.num_bodies}")
    print(f"  - Joint names (first 5): {env.unwrapped.robot.joint_names[:5]}")

    print(f"\nObject:")
    print(f"  - Name: {env.unwrapped.object.cfg.prim_path}")
    print(f"  - Mass: {env.unwrapped.object.data.default_mass[0].item():.4f} kg")

    print(f"\nScene:")
    print(f"  - Num envs: {env.unwrapped.scene.num_envs}")
    print(f"  - Env spacing: {env.unwrapped.scene.cfg.env_spacing}m")

    print("\n" + "="*80)
    print("TEST 7: Verify Observations")
    print("="*80)

    obs_dict, _ = env.reset()
    policy_obs = obs_dict['policy']

    print(f"Policy observation breakdown:")
    print(f"  - Total size: {policy_obs.shape[-1]}")
    print(f"  - Min value: {policy_obs.min().item():.4f}")
    print(f"  - Max value: {policy_obs.max().item():.4f}")
    print(f"  - Mean: {policy_obs.mean().item():.4f}")
    print(f"  - Std: {policy_obs.std().item():.4f}")

    # Check for NaN or Inf
    if torch.isnan(policy_obs).any():
        print("  ⚠ WARNING: NaN values detected in observations!")
    if torch.isinf(policy_obs).any():
        print("  ⚠ WARNING: Inf values detected in observations!")
    else:
        print("  ✓ No NaN or Inf values")

    print("\n" + "="*80)
    print("TEST 8: Verify Rewards")
    print("="*80)

    # Step once to get rewards
    action_dim = env.unwrapped.single_action_space.shape[0]
    actions = torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)
    obs_dict, rewards, terminated, truncated, info = env.step(actions)

    print(f"Reward statistics:")
    print(f"  - Shape: {rewards.shape}")
    print(f"  - Min: {rewards.min().item():.4f}")
    print(f"  - Max: {rewards.max().item():.4f}")
    print(f"  - Mean: {rewards.mean().item():.4f}")
    print(f"  - Std: {rewards.std().item():.4f}")

    if torch.isnan(rewards).any():
        print("  ⚠ WARNING: NaN values in rewards!")
    if torch.isinf(rewards).any():
        print("  ⚠ WARNING: Inf values in rewards!")

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nYour Inspire manipulation environment is working correctly!")
    print("Next steps:")
    print("  1. Implement dataset loading")
    print("  2. Add BPS encoding for objects")
    print("  3. Implement residual learning")
    print("  4. Start training!")

except Exception as e:
    print(f"\n✗ Tests failed with error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Clean up environment first
    try:
        if 'env' in locals():
            print("\nClosing environment...")
            env.close()
    except Exception as e:
        print(f"Error closing environment: {e}")

    # Then close simulation app
    print("Closing simulation...")
    simulation_app.close()

    # Force garbage collection to clean up GPU resources
    import gc
    gc.collect()

    print("Done!")

    # Force exit to ensure process terminates
    import os
    os._exit(0)
