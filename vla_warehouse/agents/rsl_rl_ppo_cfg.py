"""PPO config for the Warehouse VLA env.

Identical to vla/agents/rsl_rl_ppo_cfg.py except experiment_name, so logs
and checkpoints land in a separate logs/rsl_rl/vla_drone_warehouse/ tree.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class VLAWarehousePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 4
    max_iterations = 5000
    save_interval = 100
    experiment_name = "vla_drone_warehouse"
    logger = "tensorboard"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )

    aux_loss_weight: float = 5.0
    aux_warmup_end: int = 1000000
    aux_decay_end: int = 1000000
    aux_min_weight: float = 5.0

    lstm_hidden_dim: int = 256
    lstm_num_layers: int = 1
    lstm_type: str = "lstm"

    lora_learning_rate: float = 1.0e-6
    lora_mini_batch_size: int = 4
    lora_max_grad_norm: float = 0.5
    lora_update_every_n: int = 1
    lora_warmup_iterations: int = 50

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


# Alias so sys.modules-patched vla.agents.rsl_rl_ppo_cfg resolves
VLADronePPORunnerCfg = VLAWarehousePPORunnerCfg
