# VLA System: Vision-Language-Action Drone Navigation

## 1. System Overview

The VLA system is **Stage 3** of a three-stage curriculum for language-grounded drone navigation. It replaces frozen CLIP/SigLIP embeddings with **PaliGemma 3B** (a vision-language model), enabling the Crazyflie quadcopter to navigate toward objects based on natural language commands using raw RGB input.

### Architecture Pipeline

```
RGB (4 cameras × 224×224×3) + Text Command
                │
                ▼
    ┌───────────────────────┐
    │  PaliGemma 3B (frozen │
    │  + LoRA on q_proj,    │
    │    v_proj)             │
    └───────────┬───────────┘
                │  1024 image tokens + 24 text tokens (each 2048-dim)
                ▼
    ┌───────────────────────┐
    │  Cross-Attention Head  │
    │  (8 heads, 256-dim)    │
    │                        │
    │  Query: text embeddings│
    │  Key/Value: image embs │
    │  + 2D pos encoding     │
    │  + camera direction    │
    │  + depth grounding     │
    └───────┬───────┬───────┘
            │       │
      WHAT  │       │  WHERE
  (scene    │       │  (attention-weighted
  summary   │       │   3D ray projection)
  256-dim)  │       │  (3-dim body frame)
            ▼       ▼
    ┌───────────────────────┐
    │  LSTM Temporal Memory  │
    │  input: scene(256) +   │
    │    spatial(3) +        │
    │    flight_state(9)     │
    │  hidden: 128-dim       │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Target Prediction MLP │
    │  128 → 64 → 3          │
    │  tanh × 3.0 (meters)   │
    └───────────┬───────────┘
                │  predicted target (3-dim body frame)
                ▼
    ┌───────────────────────┐
    │  Frozen Waypoint Policy│
    │  (from Stage 2)        │
    │                        │
    │  Input: flight_state(9)│
    │    + target(3)         │
    │    + pos_error(3)      │
    │  MLP: 15→256→256→4     │
    │  Activation: ELU       │
    └───────────┬───────────┘
                │
                ▼
        4-dim action output
    [thrust, moment_x, moment_y, moment_z]
```

### Key Design Principle

The system decomposes the problem hierarchically:
- **PaliGemma** understands the visual scene and language command
- **Cross-attention** determines WHAT object and WHERE it is
- **LSTM** stabilizes predictions over time
- **Target MLP** predicts a 3D target in the drone's body frame
- **Frozen waypoint policy** handles low-level flight control (already learned in Stage 2)

---

## 2. PaliGemma 3B + LoRA Integration

**Source**: `vla_policy.py`

### Model Setup
- **Base model**: `google/paligemma-3b-pt-224` (224×224 image input)
- **All base parameters are frozen** — only LoRA adapters are trainable
- **Feature dimension**: 2048 (PaliGemma's hidden size)

### LoRA Adapter

A custom `LoRALinear` layer (no PEFT dependency):
- **Rank**: 8
- **Alpha**: 16.0, **Scale**: alpha / rank = 2.0
- **Applied to**: `q_proj` and `v_proj` layers in the transformer
- **Computation**: `output = frozen_linear(x) + lora_B(lora_A(x)) * scale`
- **Initialization**: `lora_A` with Kaiming uniform, `lora_B` with zeros (starts as identity)
- **Total trainable params**: ~16K (vs 3B frozen)

### Image Preprocessing
1. Input: `(N, H, W, 3)` float in [0, 1]
2. Permute NHWC → NCHW
3. Bilinear interpolation to 224×224
4. SigLIP normalization: `(x - 0.5) / 0.5` → [-1, 1]
5. Cast to fp16

### Feature Extraction
1. Concatenate 256 `<image>` placeholder tokens + tokenized text (~24 tokens) → 280 total tokens
2. Run PaliGemma forward pass with attention mask
3. Extract **token-level features**: `(B, 280, 2048)` for each camera
4. Image tokens: first 256, text tokens: remaining 24

### Token Caching
- During rollout, computed tokens are cached in fp16: `(B, 1048, 2048)`
  - 1048 = 4 cameras × 256 image tokens + 24 text tokens
- During PPO update, cached tokens are reused instead of re-running PaliGemma
- Memory: ~4.3 GB for 256 environments (vs 8.5 GB without fp16)

---

## 3. Four-Camera Setup

**Source**: `vla_drone_env.py`

### Camera Configuration

Four onboard cameras provide 360-degree coverage:

| Camera | Direction | Quaternion Offset | Body Axis |
|--------|-----------|-------------------|-----------|
| Front  | Forward   | (0.5, 0.5, -0.5, -0.5) | +X |
| Right  | Rightward | (0.0, 0.707, 0.0, -0.707) | +Y |
| Back   | Backward  | (0.5, -0.5, 0.5, 0.5) | -X |
| Left   | Leftward  | (0.0, 0.707, 0.0, 0.707) | -Y |

**Specs**:
- Resolution: 224×224
- Data: RGB + depth (`distance_to_camera`)
- Body offset: (0.05, 0.0, 0.01) from drone center
- FOV: ~90° (focal_length=10, horizontal_aperture=20)

### Capture Frequency
- Physics: 100 Hz
- Environment step: 50 Hz (decimation=2)
- Camera capture: every 4 env steps → **12.5 Hz**
- Cached between captures in `_cached_rgb` and `_cached_depth`

---

## 4. Cross-Attention Scene Summary

**Source**: `vla_policy.py`

The cross-attention head fuses visual tokens with language tokens to produce two outputs:
- **WHAT**: A 256-dim semantic scene summary
- **WHERE**: A 3D body-frame spatial estimate of the target

### Step-by-Step Process

#### 1. Token Projection
```
image_emb = image_proj(image_tokens)    # (B, 1024, 2048) → (B, 1024, 256)
text_emb  = text_proj(text_tokens)      # (B, 24, 2048)   → (B, 24, 256)
```

#### 2. Positional Encoding
Each image token gets three additive encodings:
- **2D sinusoidal**: 16×16 grid position (row + col), repeated for 4 cameras
- **Camera direction**: Learnable 4-dim embedding (front/right/back/left)

```
image_emb = image_emb + pos_encoding_2d + camera_embed
```

#### 3. Multi-Head Cross-Attention
```
Query:     text_emb      (B, 24, 256)
Key/Value: image_emb     (B, 1024, 256)
Heads:     8
Output:    fused          (B, 24, 256)
           attn_weights   (B, 24, 1024)
```

#### 4. Scene Summary (WHAT)
Mean-pool the fused output over valid (non-padding) text tokens:
```
scene_summary = (fused * text_mask).sum(dim=1) / text_mask.sum()   → (B, 256)
```

#### 5. Spatial Estimate (WHERE)

**a) Average attention per camera**:
```
avg_attn → (B, 1024)  →  reshape to (B, 4, 256) per camera
```

**b) Attention-weighted patch coordinates**:
```
cam_rows  = (cam_attn * patch_rows).sum()    → (B, 4)  normalized [0,1]
cam_cols  = (cam_attn * patch_cols).sum()    → (B, 4)  normalized [0,1]
cam_depth = (cam_attn * patch_depths).sum() * 20.0  → (B, 4) meters
```

**c) 3D ray construction in body frame**:
```
ray_h = (col - 0.5) * 2.0       # horizontal offset
ray_v = (row - 0.5) * 2.0       # vertical offset
ray = forward_dir + ray_h * right - ray_v * up   # normalized
body_point = ray * depth_m                         # 3D position
```

Camera forward directions: Front=[1,0,0], Right=[0,1,0], Back=[-1,0,0], Left=[0,-1,0]

**d) Weighted average across cameras**:
```
attn_spatial = (body_points * cam_mass).sum(dim=1)  → (B, 3)
```

#### 6. Object Classification
A separate MLP head from the scene summary:
```
scene_summary (256) → 64 (ELU) → 3 (logits)
```
Predicts which of the 3 objects (cube/sphere/cylinder) the command refers to.

### Depth Pooling
Raw depth images are pooled to match the 16×16 patch grid:
```
(B, 4, 224, 224) → AvgPool2d(14, 14) → (B, 4, 16, 16) → (B, 1024)
```

---

## 5. LSTM Temporal Memory

**Source**: `vla_policy.py`

### Purpose
Accumulates spatial context across timesteps for stable target prediction — prevents jittery target estimates from single-frame observations.

### Configuration
- Input: `scene_summary(256) + attn_spatial(3) + flight_state(9)` = **268-dim**
- Hidden size: 128
- Layers: 1
- Batch first: True

### State Management
- Hidden/cell states maintained across steps during rollout
- **Reset on episode done**: `_lstm_h[:, done_ids] = 0`, `_lstm_c[:, done_ids] = 0`
- **Force reset during PPO update**: Observations are shuffled across minibatches, so temporal context is invalid

---

## 6. Frozen Waypoint Policy

**Source**: `vla_policy.py`

The low-level flight controller from Stage 2 is loaded and frozen as buffer tensors (not `nn.Module` submodules).

### Loaded Weights
From checkpoint `logs/model_2998_waypoint.pt`:
- `wp_w0, wp_b0`: Layer 1 (15 → 256)
- `wp_w1, wp_b1`: Layer 2 (256 → 256)
- `wp_w2, wp_b2`: Output (256 → 4)
- `wp_obs_mean, wp_obs_std`: Frozen running normalizer

### Forward Pass
```
Input: (B, 15)
  [0:9]   = flight_state (lin_vel, ang_vel, gravity)
  [9:12]  = target_pos_body (from Target MLP)
  [12:15] = pos_error_world

Normalize: x = (obs - mean) / (std + eps)
Layer 1:   x = ELU(W0 @ x + b0)
Layer 2:   x = ELU(W1 @ x + b1)
Output:    action = W2 @ x + b2   → (B, 4)
```

---

## 7. Observation & Action Spaces

**Source**: `vla_drone_env.py`

### Observation Space (dict)

| Key | Shape | Description |
|-----|-------|-------------|
| `policy` | (N, 9) | Flight state: lin_vel(3), ang_vel(3), gravity(3) |
| `rgb` | (N, 4, 224, 224, 3) | Raw camera RGB, float [0, 1] |
| `text_tokens` | (N, 280) | Tokenized command (256 image + ~24 text tokens) |
| `text_mask` | (N, 280) | Attention mask for tokens |
| `vla_token_features` | (N, 1048, 2048) | Cached PaliGemma token features (fp16) |
| `target_gt_body` | (N, 3) | Ground-truth target in body frame (for aux loss) |
| `pos_error_w` | (N, 3) | World-frame position error |
| `target_obj_idx` | (N,) | Object class index (0=cube, 1=sphere, 2=cylinder) |
| `depth` | (N, 4, 224, 224) | Normalized depth [0, 1] |

### Action Space
4-dim continuous: `[thrust, moment_x, moment_y, moment_z]` (normalized)

---

## 8. Reward System & Navigation Curriculum

**Source**: `vla_drone_env.py`

### Reward Components

| Reward | Scale | Gating | Description |
|--------|-------|--------|-------------|
| `ang_vel` | -0.005 | Always | Penalizes angular velocity |
| `uprightness` | 0.2 | Always | Penalizes tilt |
| `altitude_penalty` | — | Always | Penalizes z < 0.3m or z > 2.8m |
| `distance_to_goal` | 35.0 | Nav, loose | Tanh-mapped distance reward |
| `velocity_toward_goal` | 4.0 | Nav, loose | Rewards movement toward target |
| `proximity` | 8.0 | Nav, loose | Bonus for entering 1.5m radius |
| `hover_at_target` | 30.0 | Nav, precise | Hovering reward with dwell bonus |
| `success` | 25.0 | Nav, precise | Bonus for < 0.35m from target |
| `wrong_object` | -3.0 | Nav, precise | Penalty for reaching wrong object |

### Two-Phase Navigation Curriculum

**Phase 1 — Learn to Navigate** (early training):
- `nav_multiplier`: ramps 0 → 1 immediately
- `loose_scale` = 1.0 (full distance/velocity/proximity rewards)
- Focus: Get the drone moving toward targets

**Phase 2 — Learn Precision** (after ~200 iterations):
- Starts at step 409,600 (`precision_curriculum_start`)
- Ramps over 614,400 steps (`precision_curriculum_steps`)
- `loose_scale`: 1.0 → 0.2 (distance rewards fade)
- `precise_scale`: 1.0 → 3.0 (hover/success rewards amplify)
- Focus: Stop and hover at the correct target

### Hover Dwell Mechanics
- **Dwell counter** increments while: `dist < hover_radius` AND `speed < 1.0 m/s`
- **Dwell reward multiplier**: `1.0 + 3.0 × (dwell / max_dwell)`
- **Max dwell**: 50 steps (1 second at 50 Hz)
- **Max multiplier**: 4.0×

### Termination Conditions
- **Success**: distance to correct target < 0.35m
- **Wrong object**: distance to wrong target < 0.35m
- **Fall**: altitude < 0.1m or > 3.0m
- **Timeout**: episode length ≥ max (15 seconds)

---

## 9. Three-Optimizer Training Strategy

**Source**: `train.py`

### Optimizers

| Optimizer | Parameters | Learning Rate | Purpose |
|-----------|-----------|---------------|---------|
| PPO | Critic + log_std | 1.0e-5 | Standard RL value/policy update |
| Auxiliary | Image/text projections, cross-attention, target MLP, object classifier, LSTM | 3.0e-4 | Direct supervision on target & object |
| LoRA | LoRA adapters only (`lora_A`, `lora_B`) | 1.0e-6 | Fine-tune PaliGemma backbone |

### Training Loop

**Rollout Phase**:
1. Step environment, capture RGB + depth every 4 steps
2. Actor forward: PaliGemma → cross-attention → LSTM → target MLP → frozen waypoint → action
3. Cache token features in fp16 for PPO replay
4. Store transitions in rollout buffer

**PPO Update Phase** (two-step):
```
Step 1: PPO loss = surrogate_loss + value_coef * value_loss - entropy_coef * entropy
        → Backward → PPO optimizer step

Step 2: aux_loss = MSE(predicted_target, gt_target) + CE(predicted_obj, gt_obj)
        → Backward → Auxiliary optimizer step
```

**LoRA Update** (after warmup of 50 iterations):
1. Sample small mini-batch (`lora_bs = min(4, num_envs)`)
2. Clear token cache to force fresh PaliGemma forward with gradients
3. Compute MSE + CE loss on the mini-batch
4. Backward through LoRA adapters only
5. LoRA optimizer step

### Auxiliary Loss Annealing
- **Iterations 0–500**: Full auxiliary loss weight
- **Iterations 500–2000**: Linear decay
- **After 2000**: Residual weight of 0.5

---

## 10. Weight Transfer from Waypoint Stage

**Source**: `transfer_waypoint_to_vla.py`

### Problem
- **Waypoint policy** input: 15-dim `[flight_state(9) + target(3) + pos_error(3)]`
- **VLA policy** input: 2057-dim `[PaliGemma_features(2048) + flight_state(9)]`

### Solution: Selective First-Layer Transfer

```
Waypoint first layer: (256 hidden, 15 input)
VLA first layer:      (256 hidden, 2057 input)

New weight matrix (256, 2057):
  [0:2048]    ← Kaiming uniform random init (PaliGemma feature slots)
  [2048:2057] ← Copy from waypoint weights[:, 0:9] (flight state slots)
```

- **Hidden and output layers**: Copied directly from waypoint checkpoint
- **Normalizer**: Only first 9 dimensions (flight state) transferred

### Usage
```bash
python vla/transfer_waypoint_to_vla.py \
    --waypoint_checkpoint logs/rsl_rl/waypoint_nav/.../model_2998.pt \
    --output_path logs/vla_init.pt
```

---

## 11. Scene & Task Setup

### Scene Objects
Three colored objects at fixed positions:
- **Red cube**
- **Blue sphere**
- **Green cylinder**

### Language Commands
- Randomly sampled per episode from a set of templates per object type
- Prepended with `"\n"` before tokenization
- Examples: "fly to the red cube", "go to the blue sphere"

### Text Tokenization Flow
1. Select random command for sampled object
2. Prepend `"\n"` + tokenize with PaliGemma tokenizer (max 24 text tokens)
3. Prepend 256 `<image>` placeholder token IDs
4. Total sequence: 280 tokens with attention mask

### Success Criteria
- Drone must be within **0.35m** of the **correct** target object
