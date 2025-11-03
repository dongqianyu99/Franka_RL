import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod

class Inspire(DexHand, ABC):
    """
    Inspire Hand base clase (12 DOFs)
    - 2 thumb joints (yaw + pitch)
    - 2 joints per finger (4 fingers × 2 = 8)
    - 2 additional thumb joints (intermediate + distal)
    """
    def __init__(self):
        super().__init__()
        self._usd_path = None
        self.side = None
        self.name = "inspire"

        # Body name
        self.body_names = [
            "hand_base_link",
            "index_proximal",
            "index_intermediate",
            "index_tip",
            "middle_proximal",
            "middle_intermediate",
            "middle_tip",
            "pinky_proximal",
            "pinky_intermediate",
            "pinky_tip",
            "ring_proximal",
            "ring_intermediate",
            "ring_tip",
            "thumb_proximal_base",
            "thumb_proximal",
            "thumb_intermediate",
            "thumb_distal",
            "thumb_tip",
        ]

        # DOF names (12 DOFs total)
        self.dof_names = [
            "index_proximal_joint",
            "index_intermediate_joint",
            "middle_proximal_joint",
            "middle_intermediate_joint",
            "pinky_proximal_joint",
            "pinky_intermediate_joint",
            "ring_proximal_joint",
            "ring_intermediate_joint",
            "thumb_proximal_yaw_joint",
            "thumb_proximal_pitch_joint",
            "thumb_intermediate_joint",
            "thumb_distal_joint",
        ]

        # Wrist is the base
        self.wrist_name = "hand_base_link"

        # Contact bodies for force sensing (5 fingertips)
        self.contact_body_names = [
            "thumb_distal",
            "index_intermediate",
            "middle_intermediate",
            "ring_intermediate",
            "pinky_intermediate",
        ]

        # Initial state configuration
        # Posision: 0.5m above ground, rotated to face down
        self.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(-0.707, 0.0, 0.0, 0.707), # -90 deg aroud X-axis
            joint_pos={
                ".*": 0.0, # All joints start at 0
            },
        )

        # Actuator configuration
        # These values control the physics simulation
        self.actuators = {
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*joint"], # Match call joint names
                effort_limit={
                    ".*": 10.0, # 10N effort limit
                },
                stiffness={
                    ".*": 500.0 # Position control stiffness
                },
                damping={
                    ".*": 30.0 # Velocity damping
                },
            ),
        }

        # Joint limits (12 DOFs, typical values for Inspire hand)
        self.dof_limit = [
            [0.0, 1.7],    # index_proximal
            [0.0, 1.7],    # index_intermediate
            [0.0, 1.7],    # middle_proximal
            [0.0, 1.7],    # middle_intermediate
            [0.0, 1.7],    # pinky_proximal
            [0.0, 1.7],    # pinky_intermediate
            [0.0, 1.7],    # ring_proximal
            [0.0, 1.7],    # ring_intermediate
            [-0.1, 1.3],   # thumb_yaw
            [0.0, 0.5],    # thumb_pitch
            [0.0, 0.8],    # thumb_intermediate
            [0.0, 1.2],    # thumb_distal
        ]

        # Weight indices for grouping joints in training
        # This is used for reward shaping
        self.weight_idx = {
            "thumb_tip": [17],
            "index_tip": [3],
            "middle_tip": [6],
            "ring_tip": [12],
            "pinky_tip": [9],
            "level_1_joints": [1, 4, 14],
            "level_2_joints": [2, 5, 7, 8, 10, 11, 13, 15, 16],
        }

        # Retargeting configuration for MABO -> Inspire hand
        self.retargeting_cfg = {
            "type": "position",
            "urdf_path": "",  # Will be set by subclass
            "target_joint_names": None,
            "target_link_names": [
                "thumb_tip",           # MANO index 4
                "index_tip",           # MANO index 8
                "middle_tip",          # MANO index 12
                "ring_tip",            # MANO index 16
                "pinky_tip",           # MANO index 20
                "thumb_intermediate",  # MANO index 2
                "index_intermediate",  # MANO index 6
                "middle_intermediate", # MANO index 10
                "ring_intermediate",   # MANO index 14
                "pinky_intermediate",  # MANO index 18
            ],
            "target_link_human_indices": [4, 8, 12, 16, 20, 2, 6, 10, 14, 18],
            "add_dummy_free_joint": True,
            "low_pass_alpha": 1,
        }

    def __str__(self):
        return self.name
    
# Register right-hand varient
@register_dexhand("inspire_rh")
class InspireRH(Inspire):
    """Inspire Right Hand"""
    def __init__(self):
        super().__init__()

        # Set USD path (if you have converted URDF to USD)
        # If not, leave as None and you'll need to handle URDF loading
        self._usd_path = None

        self._urdf_path = "assets/inspire_hand/inspire_hand_right.urdf"

        self.side = "right"

        # Add R_ prefix to all body names
        self.body_names = ["R_" + name for name in self.body_names]
        self.dof_names = ["R_" + name for name in self.dof_names]
        self.wrist_name = ["R_" + name for name in self.contact_body_names]

        # Update retargeting config
        self.retargeting_cfg["urdf_path"] = self._urdf_path
        self.retargeting_cfg["target_link_names"] = [
            "R_" + name for name in self.retargeting_cfg["target_link_names"]
        ]

        # Update actuator expressions to match R_ prefix
        self.actuators = {
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["R_.*joint"],
                effort_limit={".*":10.0},
                stiffness={".*": 500.0},
                damping={".*": 30.0},
            ),
        }

        # Update init_state joint patterns
        self.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(-0.707, 0.0, 0.0, 0.707),
            joint_pos={
                "R_.*": 0.0,
            },
        )

    def __str__(self):
        return super().__str__() + "_rh"
    
# Register left-hand varient
@register_dexhand("inspire_lh")
class InspireLH(Inspire):
    """Inspire Left Hand"""
    def __init__(self):
        super().__init__()

        self._usd_path = None
        self._urdf_path = "assets/inspire_hand/inspire_hand_left.urdf"

        self.side = "left"

        # Add L_ prefix to all body names
        self.body_names = ["L_" + name for name in self.body_names]
        self.dof_names = ["L_" + name for name in self.dof_names]
        self.wrist_name = ["L_" + name for name in self.contact_body_names]

        # Update retargeting config
        self.retargeting_cfg["urdf_path"] = self._urdf_path
        self.retargeting_cfg["target_link_names"] = [
            "L_" + name for name in self.retargeting_cfg["target_link_names"]
        ]

        # Update actuator expressions to match R_ prefix
        self.actuators = {
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["L_.*joint"],
                effort_limit={".*":10.0},
                stiffness={".*": 500.0},
                damping={".*": 30.0},
            ),
        }

        # Update init_state joint patterns
        self.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(-0.707, 0.0, 0.0, 0.707),
            joint_pos={
                "L_.*": 0.0,
            },
        )

    def __str__(self):
        return super().__str__() + "_lh"