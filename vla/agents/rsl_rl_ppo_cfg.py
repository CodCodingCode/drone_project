from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class VLADronePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 8  # reduced for faster iterations with PaliGemma 3B
    max_iterations = 5000
    save_interval = 100
    experiment_name = "vla_drone_direct"
    logger = "wandb"
    wandb_project = "drone-vla"

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

    # LoRA fine-tuning
    lora_learning_rate: float = 1.0e-6
    lora_mini_batch_size: int = 4
    lora_max_grad_norm: float = 0.5
    lora_update_every_n: int = 1
    lora_warmup_iterations: int = 50

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
