import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

import torch

@configclass
class InspireManipEnvCfg(DirectRLEnvCfg):
    """Configuration for single-hand Inspire manipulation task"""
    # ========================================================================
    # Environment Settings
    # ========================================================================

    # Episode settings
    episode_length_s = 20.0 # 20 seconds (1200 steps @ 60Hz)
    decimation = 1 # No decimation (60Hz control)

    # Action/Observation spaces
    # Action: 6D wrist pose (3 pos + 3 rot) + 12 DOF joint angles = 18
    action_space = 18
    # Observation: Will be computed in environment
    observation_space = 128  # Placeholder, actual size computed dynamically

    # Scaling
    action_scale = 1.0
    translation_scale = 0.1  # Scale for wrist translation
    orientation_scale = 0.1  # Scale for wrist rotation

    # Action smoothing
    actions_moving_average = 0.4  # Blend 40% with previous action

    # Randomization
    friction_randomization = True
    gravity_randomization = True
    friction_range = [1.0, 6.0]  # Friction coefficient range
    gravity_range = [-11.0, -9.0]  # Gravity magnitude range

    # Dataset settings
    dataset_path = "./data"  # Path to trajectory dataset
    dataset_name = "grab"  # Dataset type: "grab", "oakink", etc.
    data_indices = ["g0"]  # Which trajectories to use

    # State initialization
    random_state_init = True  # Random initial states from dataset
    rollout_state_init = False  # Sequential rollout from trajectory

    # Residual learning
    use_residual = True  # Enable residual learning
    base_model_checkpoint = ""  # Path to base model (imitator)

    # Observation future frames
    obs_future_length = 5  # Number of future target frames in obs

    # ========================================================================
    # Simulation Settings
    # ========================================================================

    sim: SimulationCfg = SimulationCfg(
        dt=1/60.0,  # 60Hz simulation
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=1,  # TGS solver (more stable)
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**20,  # Increase for complex contacts
            gpu_max_rigid_patch_count=2**19,
            gpu_found_lost_pairs_capacity=2**20,
            gpu_found_lost_aggregate_pairs_capacity=2**16,
            gpu_total_aggregate_pairs_capacity=2**16,
        ),
    )

    # ========================================================================
    # Scene Settings
    # ========================================================================

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,  # 4096 parallel environments
        env_spacing=2.0,  # 2m spacing between environments
        replicate_physics=False,  # Share physics across envs
    )

    # Terrain (ground plane)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,  # Will be randomized during training
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ========================================================================
    # Robot Configuration (Inspire Hand)
    # ========================================================================

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/InspireHand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="assets/inspire_hand/inspire_hand_right.urdf",
            fix_base=True,  # Fix the wrist in place (we'll control it separately)
            activate_contact_sensors=True,  # Enable contact sensing
            # Joint drive configuration for URDF conversion
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=500.0,  # Default stiffness for all joints
                    damping=30.0,    # Default damping for all joints
                ),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3),  # Start 30cm above ground
            rot=(0.0, 0.0, 0.0, 1.0),  # Identity quaternion
            joint_pos={
                "R_.*": 0.0,  # All joints at 0
            },
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["R_.*joint"],
                effort_limit=10.0,
                stiffness=500.0,
                damping=30.0,
            ),
        },
    )

    # ========================================================================
    # Object Configuration
    # ========================================================================

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",  # Will be set from dataset
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # Will be set from dataset
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),  # 10cm above ground
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # ========================================================================
    # Reward Weights (from ManipTrans)
    # ========================================================================

    # Distance reward: Fingertip to object distance
    reward_distance_weight = 1.0
    reward_distance_scale = 10.0  # Exponential scale

    # Lift reward: Object height above ground
    reward_lift_weight = 0.5
    reward_lift_threshold = 0.05  # 5cm lift threshold

    # Success reward: Object lifted and stable
    reward_success_weight = 10.0
    reward_success_threshold = 0.1  # 10cm lift for success

    # Action smoothness penalty
    reward_action_smooth_weight = 0.01

    # Joint limit penalty
    reward_joint_limit_weight = 0.0001

    # Contact reward: Encourage contact with object
    reward_contact_weight = 0.1

    # ========================================================================
    # BPS (Basis Point Set) Configuration
    # ========================================================================

    # BPS encoding for object shape
    bps_num_points = 128  # Number of BPS basis points
    bps_radius = 0.3  # BPS sphere radius (30cm)

    # ========================================================================
    # Logging and Debugging
    # ========================================================================

    log_train_info = False
    info_buffer_size = 500000
    log_dir = "./log_info/"