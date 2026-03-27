# Language-Grounded Drone Navigation

This project trains a quadcopter drone to navigate to objects based on natural language commands using reinforcement learning in simulation.

A user gives a command like *"fly to the ball"* or *"go to the cube"*, and the drone learns to identify the correct target and fly to it — all from language alone, no explicit goal coordinates.

## How It Works

The system places a **Crazyflie quadcopter** in a simulated arena containing three colored objects (a red cube, a blue sphere, and a green cylinder). Each episode, a random language command is sampled (e.g. *"navigate to the pillar"*), encoded into a 512-dimensional embedding using a frozen **OpenAI CLIP** model, and fed to the drone's policy network alongside its flight state and the relative positions of all objects.

A PPO (Proximal Policy Optimization) agent learns to output thrust and moment commands that steer the drone to the correct target. Training runs across **1024 parallel environments** on GPU using NVIDIA Isaac Sim and Isaac Lab.

### Reward Design

| Signal | Value | Purpose |
|---|---|---|
| Distance shaping | scaled by 15.0 | Guides the drone toward the correct object |
| Velocity penalty | negative | Encourages smooth, stable flight |
| Correct target reached | +5.0 | Success bonus (within 0.35 m) |
| Wrong target reached | -3.0 | Discourages going to the wrong object |

## Tech Stack

- **NVIDIA Isaac Sim + Isaac Lab** — GPU-accelerated physics simulation and RL environment framework
- **Pegasus Simulator** — flight dynamics for multirotor drones
- **RSL-RL** — PPO implementation from ETH Zurich's Robotics Systems Lab
- **OpenAI CLIP** (`clip-vit-base-patch32`) — frozen text encoder for grounding language commands
- **PyTorch** — neural network training

## Project Structure

```
drone_project/
├── main.py                       # Standalone sim modes (hover test, scene viewer)
└── lang_nav/
    ├── __init__.py               # Gymnasium env registration
    ├── train.py                  # Training entry point (launched via isaaclab.sh)
    ├── lang_drone_env.py         # RL environment definition
    ├── clip_grounder.py          # CLIP text encoder wrapper
    ├── commands.py               # Command bank (object → phrase variants)
    └── agents/
        └── rsl_rl_ppo_cfg.py     # PPO hyperparameters
```

## Running Training

```bash
cd ~/IsaacLab
./isaaclab.sh -p ~/drone_project/lang_nav/train.py \
    --num_envs 1024 --max_iterations 500 --headless
```

Checkpoints and TensorBoard logs are saved to `drone_project/logs/rsl_rl/lang_drone_direct/`.

## Policy Architecture

The actor and critic are both MLPs (533 → 256 → 256 → output) with ELU activations and observation normalization. The 533-dim observation is composed of:

- **12 dims** — drone velocity and projected gravity
- **512 dims** — CLIP text embedding of the current command
- **9 dims** — relative positions of the three target objects
