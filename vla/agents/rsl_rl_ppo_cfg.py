from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class VLADronePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16  # reduced from 24 for GPU memory (images in rollout)
    max_iterations = 5000
    save_interval = 100
    experiment_name = "vla_drone_direct"
    logger = "wandb"
    wandb_project = "drone-vla"

    # NOTE: actor and critic are custom nn.Modules, not RslRlMLPModelCfg.
    # They are constructed manually in train.py, not via construct_algorithm.

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
