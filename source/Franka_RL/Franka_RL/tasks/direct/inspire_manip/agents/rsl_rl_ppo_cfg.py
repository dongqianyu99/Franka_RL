from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class InspireManipPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for PPO training with Ispire hand."""

    # ========================================================================
    # Training Settings
    # ========================================================================

    # Number of steps to collect per environment before each policy update
    num_steps_per_env = 32 # Horizon length

    # Maximum number of training iterations
    max_iterations = 10000

    # Save checkpoint every N iterations
    save_interval = 100

    # Logging interval
    log_inteval = 10

    # Use empirical normalization for observations
    empirical_normalization = True

    # ========================================================================
    # Policy Network Configuration
    # ========================================================================

    policy = RslRlPpoActorCriticCfg(
        # Initialize the policy class
        # We'll use the default MLP architecture
        class_name="ActorCritic",
        
        # Network dimensions
        init_noise_std=1.0, # Initial std for action distribution

        # Actor network (policy)
        actor_hidden_dims=[256, 512, 128, 64],

        # Critic network (value funtion)
        critic_hidden_dims=[256, 512, 128, 64],

        # Activation function
        activation="elu", # From ManipTrans (options: relu, elu,  selu, crelu, lrelu, tanh, sigmoid)
    )

    # ========================================================================
    # PPO Algorithm Configuration
    # ========================================================================

    algorithm = RslRlPpoAlgorithmCfg(
        # Discount factor
        gamma=0.99,

        # GAE lambda (Generallized Advantage Estimation)
        lam=0.95,

        # Learning rate
        learning_rate=2e-4,

        # Learning rate schedule (options: adaptive, constant, linear)
        schedule="adaptive",  # Note: parameter name is 'schedule' not 'lr_schedule'

        # Number of optimization epochs per batch
        num_learning_epochs=5,

        # Number of mini-batches per epoch
        num_mini_batches=4,

        # PPO clipping parameter
        clip_param=0.2,

        # Desired KL divergence for adaptive learning rate
        desired_kl=0.01,

        # Entropy bonus coefficient
        entropy_coef=0.0,

        # Value function loss coefficient
        value_loss_coef=4.0,

        # Maximum gradient norm for clipping
        max_grad_norm=1.0,

        # Use clipped value function loss
        use_clipped_value_loss=True,
    )

# Optional: Configuration for residual learning
@configclass  
class InspireManipResidualPPORunnerCfg(InspireManipPPORunnerCfg):
    """
    Configuration for residual learning with pre-trained base model.
    
    Inherits from base config and adds residual-specific settings.
    """

    # Path to pre-trained base model (imitator)
    base_model_checkpoint: str = ""

    # Whether to freeze base model
    freeze_base_model: bool = True

    # Residual learning might converge faster
    max_iterations = 5000

    # Lower learning rate for fine-tuning
    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=1e-4,  # Lower than base training
        gamma=0.99,
        lam=0.95,
        schedule="adaptive",
        desired_kl=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        entropy_coef=0.0,
        value_loss_coef=4.0,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
    )