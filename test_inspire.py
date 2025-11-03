import argparse

# Step 1: Import AppLauncher FIRST (before any other Isaac Lab imports)
from isaaclab.app import AppLauncher

# Step 2: Create argument parser
parser = argparse.ArgumentParser(description="Test Inspire hand registration")
# Add AppLauncher arguments (handles Isaac Sim settings)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Step 3: Launch Isaac Sim app (CRITICAL - must happen before other imports)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Step 4: NOW you can import Isaac Lab modules and your robots
print("Isaac Sim launched successfully!")
print("Importing robot modules...")

from Franka_RL.robots import RobotFactory

# Test 1: Create Inspire right hand
print("\n" + "="*60)
print("Test 1: Creating Inspire Right Hand")
print("="*60)

try:
    robot_rh = RobotFactory.create_robot("inspire_rh")
    print(f"✓ Successfully created: {robot_rh}")
    print(f"  - Name: {robot_rh.name}")
    print(f"  - Side: {robot_rh.side}")
    print(f"  - DOFs: {robot_rh.n_dofs}")
    print(f"  - Bodies: {robot_rh.n_bodies}")
    print(f"  - Joint names (first 3): {robot_rh.dof_names[:3]}")
    print(f"  - Contact bodies: {robot_rh.contact_body_names}")
except Exception as e:
    print(f"✗ Failed to create inspire_rh: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Create Inspire left hand
print("\n" + "="*60)
print("Test 2: Creating Inspire Left Hand")
print("="*60)

try:
    robot_lh = RobotFactory.create_robot("inspire_lh")
    print(f"✓ Successfully created: {robot_lh}")
    print(f"  - Name: {robot_lh.name}")
    print(f"  - Side: {robot_lh.side}")
    print(f"  - DOFs: {robot_lh.n_dofs}")
except Exception as e:
    print(f"✗ Failed to create inspire_lh: {e}")
    import traceback
    traceback.print_exc()

# Test 3: List all registered robots
print("\n" + "="*60)
print("Test 3: All Registered Robots")
print("="*60)

from Franka_RL.robots.factory import RobotFactory
print(f"Available robots: {list(RobotFactory._registry.keys())}")

# Test 4: Check actuator configuration
print("\n" + "="*60)
print("Test 4: Actuator Configuration")
print("="*60)

print(f"Actuator groups: {list(robot_rh.actuators.keys())}")
for name, actuator_cfg in robot_rh.actuators.items():
    print(f"  {name}:")
    print(f"    - Joint pattern: {actuator_cfg.joint_names_expr}")
    print(f"    - Effort limit: {actuator_cfg.effort_limit}")
    print(f"    - Stiffness: {actuator_cfg.stiffness}")
    print(f"    - Damping: {actuator_cfg.damping}")

# Test 5: Check initial state
print("\n" + "="*60)
print("Test 5: Initial State Configuration")
print("="*60)

print(f"Initial position: {robot_rh.init_state.pos}")
print(f"Initial rotation: {robot_rh.init_state.rot}")
print(f"Initial joint config: {robot_rh.init_state.joint_pos}")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)

# Step 5: Close the simulation app
print("\nClosing simulation...")
simulation_app.close()

# Force garbage collection to clean up GPU resources
import gc
gc.collect()

print("Done!")

# Force exit to ensure process terminates
import os
os._exit(0)