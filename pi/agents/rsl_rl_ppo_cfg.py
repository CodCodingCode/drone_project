from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class Pi0DronePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 8
    max_iterations = 5000
    save_interval = 100
    experiment_name = "pi0_drone_direct"
    logger = "wandb"
    wandb_project = "drone-pi0"

    # NOTE: actor and critic are custom nn.Modules, not standard RslRl models.
    # They are constructed manually in train.py, not via construct_algorithm.
    # We provide a dummy policy config to satisfy the runner's required field.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )

    # No LoRA fields -- backbone is fully frozen

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,   # Higher than VLA (1e-5) since only MLP params
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
