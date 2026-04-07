from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class VLADronePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 8  # reduced for faster iterations with PaliGemma 3B
    max_iterations = 5000
    save_interval = 100
    experiment_name = "vla_drone_direct"
    logger = "tensorboard"

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

    # Auxiliary target supervision (MSE on cross-attention head's predicted target vs ground truth)
    aux_loss_weight: float = 5.0       # initial/peak MSE weight (high because head is detached from PPO)
    aux_warmup_end: int = 500          # full weight until this iteration
    aux_decay_end: int = 2000          # linear decay to aux_min_weight
    aux_min_weight: float = 0.1        # residual weight after decay (prevents drift)

    # LSTM temporal memory
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 1
    lstm_type: str = "lstm"

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
