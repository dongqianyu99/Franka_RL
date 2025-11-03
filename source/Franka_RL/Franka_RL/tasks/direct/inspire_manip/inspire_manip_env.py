"""
Lifecycle of a DirectRLEnv:

Initialization:
    __init__()
        ↓
    _setup_scene()        # Add robot, object, ground
        ↓
    _configure_scene()    # (automatic - Isaac Lab handles this)

Training Loop (repeated):
    reset()
        ↓
    _reset_idx(env_ids)   # Reset specific environments
        ↓
    step(actions)
        ↓
    _pre_physics_step(actions)     # Process actions
        ↓
    _apply_action()                # Send to simulator
        ↓
    (Physics simulation runs)
        ↓
    _get_observations() → obs
        ↓
    _get_rewards() → rewards
        ↓
    _get_dones() → dones
"""

from __future__ import annotations

import torch
import numpy as np
import os
from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_mul, quat_conjugate

from .inspire_manip_env_cfg import InspireManipEnvCfg

# Import dataset support (we'll need to create this)
# from Franka_RL.dataset import DataFactory


class InspireManipEnv(DirectRLEnv):
    cfg: InspireManipEnvCfg

    def __init__(self, cfg: InspireManipEnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the environment

        Args:
            cfg: Environment configuration
            render_mode: Rendering mode (None, "human", "rgb_array)
            **kwargs: Additional arguments passed to DirectRLEnv
        """
        super().__init__(cfg, render_mode, **kwargs)

        # ====================================================================
        # Post-initialization Setup
        # ====================================================================

        # Action smoothing (moving average)
        self.actions_moving_average = cfg.actions_moving_average
        # Initialize previous_actions with same shape as self.actions (from base class)
        self.previous_actions = torch.zeros_like(self.actions)

        # Get joint indices for the robot
        self._joint_dof_idx, _ = self.robot.find_joints("R_.*joint")
        self.num_hand_dofs = len(self._joint_dof_idx) # 12 for Inspire

        # Action scaling
        self.action_scale = cfg.action_scale
        self.translation_scale = cfg.translation_scale
        self.orientation_scale = cfg.orientation_scale

        # Episode step counter
        self.episode_step_counter = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int32
        )

        # ====================================================================
        # Dataset Loading
        # ====================================================================

        # TODO: Uncomment when dataset is implemented
        # from Franka_RL.robots import RobotFactory
        # from Franka_RL.dataset import DataFactory
        # self.dataset = DataFactory.create_data(
        #     cfg.dataset_name,
        #     data_dir=cfg.dataset_path,
        #     dexhand=RobotFactory.create_robot("inspire_rh"),
        #     device=self.device
        # )

        # For now, we'll use dummy data
        print("[INFO] Using dummy dataset (replace with real dataset later)")
        self._init_dummy_dataset()

        # ====================================================================
        # BPS (Basis Point Set) for Object Encoding
        # ====================================================================

        # BPS is used to encode object geometry into a fixed-size feature vector
        # TODO: Initialize BPS encoder
        # from bps_torch.bps import bps_torch
        # self.bps = bps_torch(
        #     bps_type='sphere',
        #     n_bps_points=cfg.bps_num_points,
        #     radius=cfg.bps_radius,
        #     device=self.device
        # )

        # ====================================================================
        # Residual Learning
        # ====================================================================

        if cfg.use_residual and cfg.base_model_checkpoint:
            print(f"[INFO] Loading base model from {cfg.base_model_checkpoint}")
            self.base_model = self._load_base_model(cfg.base_model_checkpoint)
            self.base_model.eval()  # Set to evaluation mode
        else:
            self.base_model = None

        # ====================================================================
        # Observation/Action Buffers
        # ====================================================================

        # These will be computed in _get_observations()
        self.obs_buf_policy = None
        self.obs_buf_critic = None

        print(f"[INFO] InspireManipEnv initialized with {self.num_envs} environments")
        print(f"[INFO] Robot DOFs: {self.num_hand_dofs}")
        print(f"[INFO] Action space: {self.cfg.action_space}")

    # ========================================================================
    # SCENE SETUP
    # ======================================================================== 

    def _setup_scene(self):
        """
        Set up the simulation scene.
        
        This is called once during __init__() to create all the assets.
        Isaac Lab will automatically clone these across all environments.
        """
        print("[INFO] Setting up scene...")

        # ----------------------------------------------------------------
        # 1. Create the robot (Inspire Hand)
        # ----------------------------------------------------------------
        self.robot = Articulation(self.cfg.robot)

        # ----------------------------------------------------------------
        # 2. Create the ground plane
        # ----------------------------------------------------------------
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # ----------------------------------------------------------------
        # 3. Create the object
        # ----------------------------------------------------------------
        # Note: In real implementation, you'd spawn different objects per env
        # For now, we'll create a simple sphere as placeholder
        self.object = RigidObject(self._create_placeholder_object_cfg())

        # ----------------------------------------------------------------
        # 4. Filter collisions
        # ----------------------------------------------------------------
        # Prevent ground from colliding with itself
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # ----------------------------------------------------------------
        # 5. Register assets with scene
        # ----------------------------------------------------------------
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        # ----------------------------------------------------------------
        # 6. Add lighting
        # ----------------------------------------------------------------
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75)
        )
        light_cfg.func("/World/Light", light_cfg)

        print("[INFO] Scene setup complete")

    def _create_placeholder_object_cfg(self):
        """Create a simple sphere object as placeholder."""
        from isaaclab.assets import RigidObjectCfg

        return RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.SphereCfg(
                radius=0.05,  # 5cm sphere
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 100g
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.2, 0.2),  # Red
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.15),  # 15cm above ground
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

    # ========================================================================
    # ACTION PROCESSING
    # ========================================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Process actions before physics simulation.

        This is called every control step (before physics steps).

        Args:
            actions: Shape (num_envs, action_space)
                     [0:3]   = wrist translation (X, Y, Z)
                     [3:6]   = wirst rotation (axis-angle or euler)
                     [6:18]  = joint angles (12 DOFs)
        """
        # ----------------------------------------------------------------
        # 1. Apply action smoothing (moving average)
        # ----------------------------------------------------------------
        if self.actions_moving_average > 0:
            actions = (
                self.actions_moving_average * self.previous_actions +
                (1.0 - self.actions_moving_average) * actions
            )
            self.previous_actions = actions.clone()

        # ----------------------------------------------------------------
        # 2. Residual learning: Add base model action if enabled
        # ----------------------------------------------------------------
        if self.base_model is not None:
            with torch.no_grad():
                # Get base action from imitator
                base_actions = self._get_base_model_actions()
                # Add residual (current action) to base
                actions = base_actions + actions

        # ----------------------------------------------------------------
        # 3. Store processed actions
        # ----------------------------------------------------------------
        self.actions = actions

        # Increment episode counter
        self.episode_step_counter += 1

    def _apply_action(self):
        """
        Apply processed actions to the robot.

        This is called after _pre_physics_step() to send commands to simulator.
        """

        # ----------------------------------------------------------------
        # Extract action components
        # ----------------------------------------------------------------
        wrist_translation = self.actions[:, 0:3] * self.translation_scale
        wrist_rotation = self.actions[:, 3:6] * self.orientation_scale
        joint_targets = self.actions[:, 6:6+self.num_hand_dofs] * self.action_scale

        # ----------------------------------------------------------------
        # Apply wrist control
        # ----------------------------------------------------------------
        # Get current wrist pose
        current_wrist_pos = self.robot.data.root_pos_w
        current_wrist_quat = self.robot.data.root_quat_w

        # Apply translation (in world frame)
        target_wrist_pos = current_wrist_pos + wrist_translation

        # Apply rotation (axis-angle to quaternion)
        # Convert axis-angle to quaternion
        angle = torch.norm(wrist_rotation, dim=-1, keepdim=True)
        axis = wrist_rotation / (angle + 1e-8)
        delta_quat = self._axis_angle_to_quat(axis, angle)
        target_wrist_quat = quat_mul(delta_quat, current_wrist_quat)

        # Set wrist pose
        # Concatenate position and quaternion into single 7D tensor
        target_wrist_pose = torch.cat([target_wrist_pos, target_wrist_quat], dim=-1)
        self.robot.write_root_pose_to_sim(target_wrist_pose)

        # ----------------------------------------------------------------
        # Apply joint control
        # ----------------------------------------------------------------
        # Clamp joint targets to limits (optional, Isaac Lab handles this)
        # joint_targets = torch.clamp(joint_targets, -1.0, 1.0)

        # Set joint position targets
        # Isaac Lab's implicit actuators will convert to torques via PD control
        self.robot.set_joint_position_target(
            joint_targets,
            joint_ids=self._joint_dof_idx
        )

    # ========================================================================
    # OBSERVATION COMPUTATION
    # ========================================================================

    def _get_observations(self) -> dict:
        """
        Compute observations for the policy and critic.
        
        Returns:
            Dictionary with keys:
                - "policy": Observations for the actor (non-privileged)
                - "critic": Observations for the critic (privileged)
        """
        # ----------------------------------------------------------------
        # 1. Proprioceptive observations (robot state)
        # ----------------------------------------------------------------

        # Joint positions and velocities
        joint_pos = self.robot.data.joint_pos[:, self._joint_dof_idx]  # (N, 12)
        joint_vel = self.robot.data.joint_vel[:, self._joint_dof_idx]  # (N, 12)

        # Wrist pose (root body)
        wrist_pos = self.robot.data.root_pos_w  # (N, 3)
        wrist_quat = self.robot.data.root_quat_w  # (N, 4) - (x,y,z,w)
        wrist_lin_vel = self.robot.data.root_lin_vel_w  # (N, 3)
        wrist_ang_vel = self.robot.data.root_ang_vel_w  # (N, 3)

        # Base state: pos + quat + lin_vel + ang_vel = 3+4+3+3 = 13
        base_state = torch.cat([
            wrist_pos,
            wrist_quat,
            wrist_lin_vel,
            wrist_ang_vel
        ], dim=-1)  # (N, 13)

        # ----------------------------------------------------------------
        # 2. Object observations
        # ----------------------------------------------------------------

        # Object pose and velocity
        object_pos = self.object.data.root_pos_w  # (N, 3)
        object_quat = self.object.data.root_quat_w  # (N, 4)
        object_lin_vel = self.object.data.root_lin_vel_w  # (N, 3)
        object_ang_vel = self.object.data.root_ang_vel_w  # (N, 3)

        # Object state relative to wrist
        object_pos_rel = object_pos - wrist_pos  # (N, 3)

        # ----------------------------------------------------------------
        # 3. Fingertip positions (for distance to object)
        # ----------------------------------------------------------------

        # Get fingertip body indices
        # For Inspire hand: thumb_distal, index_intermediate, middle_intermediate, etc.
        fingertip_names = [
            "R_thumb_distal",
            "R_index_intermediate",
            "R_middle_intermediate",
            "R_ring_intermediate",
            "R_pinky_intermediate"
        ]

        fingertip_positions = []
        for name in fingertip_names:
            body_idx, _ = self.robot.find_bodies(name)
            if len(body_idx) > 0:
                pos = self.robot.data.body_pos_w[:, body_idx[0]]  # (N, 3)
                fingertip_positions.append(pos)

        fingertip_positions = torch.stack(fingertip_positions,
dim=1)  # (N, 5, 3)

        # Distance from each fingertip to object
        fingertip_to_object = fingertip_positions - object_pos.unsqueeze(1)  # (N, 5, 3)
        fingertip_distances = torch.norm(fingertip_to_object,
dim=-1)  # (N, 5)

        # ----------------------------------------------------------------
        # 4. Target observations (from dataset)
        # ----------------------------------------------------------------

        # TODO: Get future target frames from dataset
        # For now, use dummy targets
        target_features = torch.zeros(
            self.num_envs, self.cfg.bps_num_points,
            device=self.device, dtype=torch.float32
        )  # (N, 128) - BPS encoded target

        # ----------------------------------------------------------------
        # 5. Assemble policy observations (non-privileged)
        # ----------------------------------------------------------------

        obs_policy = torch.cat([
            base_state,                          # (13)
            joint_pos,                           # (12)
            object_pos_rel,                      # (3)
            fingertip_distances,                 # (5)
            target_features,                     # (128)
        ], dim=-1)  # Total: ~161

        # ----------------------------------------------------------------
        # 6. Assemble critic observations (privileged)
        # ----------------------------------------------------------------

        # Critic gets additional information
        obs_critic = torch.cat([
            obs_policy,                          # All policy obs
            joint_vel,                           # (12) - velocities
            object_lin_vel,                      # (3)
            object_ang_vel,                      # (3)
            wrist_lin_vel,                       # (3)
            wrist_ang_vel,                       # (3)
        ], dim=-1)  # Policy + ~24 extra

        # ----------------------------------------------------------------
        # Store for reward computation
        # ----------------------------------------------------------------
        self.obs_buf_policy = obs_policy
        self.obs_buf_critic = obs_critic

        # Store useful quantities for rewards
        self.fingertip_distances = fingertip_distances
        self.object_height = object_pos[:, 2]  # Z coordinate

        return {"policy": obs_policy, "critic": obs_critic}

    # ========================================================================
    # REWARD COMPUTATION
    # ========================================================================

    def _get_rewards(self) -> torch.Tensor:
        """
        Compute reward for each environment.
        
        Returns:
            Tensor of shape (num_envs,) with rewards
        """
        # ----------------------------------------------------------------
        # 1. Distance reward: Minimize fingertip-to-object distance
        # ----------------------------------------------------------------

        # Mean distance across all fingertips
        mean_distance = torch.mean(self.fingertip_distances,
dim=-1)  # (N,)

        # Exponential reward: exp(-scale * distance)
        distance_reward = torch.exp(
            -self.cfg.reward_distance_scale * mean_distance
        )
        distance_reward = self.cfg.reward_distance_weight * distance_reward

        # ----------------------------------------------------------------
        # 2. Lift reward: Encourage lifting object
        # ----------------------------------------------------------------

        # Reward for lifting above threshold
        lift_reward = torch.where(
            self.object_height > self.cfg.reward_lift_threshold,
            torch.ones_like(self.object_height),
            torch.zeros_like(self.object_height)
        )
        lift_reward = self.cfg.reward_lift_weight * lift_reward

        # ----------------------------------------------------------------
        # 3. Success reward: Large bonus for successful lift
        # ----------------------------------------------------------------

        success_reward = torch.where(
            self.object_height > self.cfg.reward_success_threshold,
            torch.ones_like(self.object_height),
            torch.zeros_like(self.object_height)
        )
        success_reward = self.cfg.reward_success_weight * success_reward

        # ----------------------------------------------------------------
        # 4. Action smoothness penalty
        # ----------------------------------------------------------------

        # Penalize large action changes
        if hasattr(self, 'previous_actions'):
            action_diff = torch.norm(
                self.actions - self.previous_actions,
                dim=-1
            )
            action_smooth_penalty = -self.cfg.reward_action_smooth_weight * action_diff
        else:
            action_smooth_penalty = torch.zeros(self.num_envs, device=self.device)

        # ----------------------------------------------------------------
        # 5. Total reward
        # ----------------------------------------------------------------

        total_reward = (
            distance_reward +
            lift_reward +
            success_reward +
            action_smooth_penalty
        )

        return total_reward
    
    # ========================================================================
    # TERMINATION LOGIC
    # ========================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Check which environments should terminate.
        
        Returns:
            Tuple of (terminated, truncated) tensors, each shape (num_envs,)
            - terminated: Episode ended due to task  completion/failure
            - truncated: Episode ended due to time limit
        """
        # ----------------------------------------------------------------
        # Time limit truncation
        # ----------------------------------------------------------------
        max_episode_steps = int(
            self.cfg.episode_length_s / (self.cfg.decimation * self.cfg.sim.dt)
        )
        truncated = self.episode_step_counter >= max_episode_steps

        # ----------------------------------------------------------------
        # Task-based termination (optional)
        # ----------------------------------------------------------------

        # Example: Terminate if object falls too far
        # object_fell = self.object_height < -0.1  # 10cm below ground

        # Example: Terminate if hand moves too far from object
        # hand_too_far = torch.mean(self.fingertip_distances, dim=-1) > 0.5

        # For now, only use time limit
        terminated = torch.zeros_like(truncated)

        return terminated, truncated
    
    # ========================================================================
    # RESET LOGIC
    # ========================================================================

    def _reset_idx(self, env_ids: torch.Tensor):
        """
        Reset specific environments.
        
        Args:
            env_ids: Indices of environments to reset
        """
        if len(env_ids) == 0:
            return

        # ----------------------------------------------------------------
        # 1. Reset episode counter
        # ----------------------------------------------------------------
        self.episode_step_counter[env_ids] = 0

        # ----------------------------------------------------------------
        # 2. Reset robot state
        # ----------------------------------------------------------------

        if self.cfg.random_state_init:
            # Random initial state (from dataset or random sampling)
            # TODO: Load from dataset

            # For now, random around initial pose
            wrist_pos = self.robot.data.default_root_state[env_ids, :3].clone()
            wrist_pos += sample_uniform(
                -0.1, 0.1,
                (len(env_ids), 3),
                device=self.device
            )
            wrist_quat = self.robot.data.default_root_state[env_ids, 3:7].clone()

            joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
            joint_pos += sample_uniform(
                -0.2, 0.2,
                joint_pos.shape,
                device=self.device
            )

        else:
            # Default initial state
            wrist_pos = self.robot.data.default_root_state[env_ids, :3]
            wrist_quat = self.robot.data.default_root_state[env_ids, 3:7]
            joint_pos = self.robot.data.default_joint_pos[env_ids]

        # Reset velocities to zero
        wrist_vel = torch.zeros((len(env_ids), 6), device=self.device)
        joint_vel = torch.zeros_like(joint_pos)

        # Write to simulation
        # Concatenate position and quaternion into single 7D tensor
        wrist_pose = torch.cat([wrist_pos, wrist_quat], dim=-1)
        self.robot.write_root_pose_to_sim(wrist_pose, env_ids)
        self.robot.write_root_velocity_to_sim(wrist_vel, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ----------------------------------------------------------------
        # 3. Reset object state
        # ----------------------------------------------------------------

        # Random object position near hand
        object_pos = torch.zeros((len(env_ids), 3), device=self.device)
        object_pos[:, 0] = sample_uniform(-0.1, 0.1, (len(env_ids),), device=self.device)
        object_pos[:, 1] = sample_uniform(-0.1, 0.1, (len(env_ids),), device=self.device)
        object_pos[:, 2] = 0.15  # 15cm above ground

        object_quat = torch.zeros((len(env_ids), 4), device=self.device)
        object_quat[:, 3] = 1.0  # Identity quaternion

        object_vel = torch.zeros((len(env_ids), 6), device=self.device)

        # Concatenate position and quaternion into single 7D tensor
        object_pose = torch.cat([object_pos, object_quat], dim=-1)
        self.object.write_root_pose_to_sim(object_pose, env_ids)
        self.object.write_root_velocity_to_sim(object_vel, env_ids)

        # ----------------------------------------------------------------
        # 4. Reset previous actions
        # ----------------------------------------------------------------
        self.previous_actions[env_ids] = 0.0

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _init_dummy_dataset(self):
        """Initialize dummy dataset for testing (replace with real dataset later)."""
        # Placeholder - just creates empty tensors
        self.dummy_targets = torch.zeros(
            self.num_envs,
            self.cfg.obs_future_length,
            7,  # 3 pos + 4 quat
            device=self.device
        )

    def _axis_angle_to_quat(self, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle to quaternion.
        
        Args:
            axis: Unit axis, shape (..., 3)
            angle: Rotation angle in radians, shape (..., 1)
        
        Returns:
            Quaternion (x,y,z,w), shape (..., 4)
        """
        half_angle = angle / 2.0
        quat = torch.zeros((*axis.shape[:-1], 4), device=axis.device, dtype=axis.dtype)
        quat[..., :3] = axis * torch.sin(half_angle)
        quat[..., 3] = torch.cos(half_angle).squeeze(-1)
        return quat

    def _load_base_model(self, checkpoint_path: str):
        """Load pre-trained base model for residual learning."""
        # TODO: Implement base model loading
        # return torch.load(checkpoint_path,  map_location=self.device)
        print("[WARNING] Base model loading not implemented yet")
        return None

    def _get_base_model_actions(self) -> torch.Tensor:
        """Get actions from base model (imitator)."""
        # TODO: Implement base model inference
        # with torch.no_grad():
        #     return self.base_model(self.obs_buf_policy)
        return torch.zeros(self.num_envs, self.cfg.action_space,device=self.device)