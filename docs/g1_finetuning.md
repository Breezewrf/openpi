# G1 23-DoF Robojudo-Plus 数据集微调指南

本文档说明如何使用下面这份本地 LeRobot v2.1 数据集训练 OpenPI：

```text
/home/breeze/Desktop/workplace/Isaac-GR00T/demo_data/g1_23dof_upper_body_pickup_mulcam_openpi
```

`_openpi` 副本已经修复 MP4 时间戳。本接口验证时，数据集包含 111 个 episode、67,736 帧、30 Hz、三路
RGB 相机、两个语言任务、30 维上肢 state 和 34 维 action。

## 已实现的适配

- `src/openpi/policies/g1_policy.py`：负责 G1 训练输入和推理输出的格式转换。
- `src/openpi/training/config.py` 中的 `LeRobotG1DataConfig`：负责 LeRobot 字段映射、action chunk 和
  absolute-to-delta 转换。
- `pi05_g1_pickup_lora`：推荐使用的低显存 JAX LoRA 配置。
- `pi05_g1_pickup`：π0.5 全量微调配置。
- `scripts/fix_lerobot_video_timestamps.py`：无损检查、修复 MP4 非零起始时间戳。

相机映射如下：

| 数据集字段 | 模型图像槽位 |
|---|---|
| `observation.images.head_rgb` | `base_0_rgb` |
| `observation.images.left_wrist_rgb` | `left_wrist_0_rgb` |
| `observation.images.right_wrist_rgb` | `right_wrist_0_rgb` |

接口沿用 Isaac-GR00T `examples/RoboJuDo/robojudo_g1_23dof_mulcam_config.py` 的语义：state 是 30 维，
action 是完整的 34 维。只有左右臂的前 10 维在训练时转成相对当前 state 的 delta；左右灵巧手、导航命令和
高度命令均保持绝对量。推理输出经过逆变换后仍是完整的 34 维 RoboJuDo action。

| action 范围 | 分组 | 维数 | 训练表示 |
|---|---|---:|---|
| `[0:5]` | `left_arm` | 5 | relative，推理后恢复为 absolute |
| `[5:10]` | `right_arm` | 5 | relative，推理后恢复为 absolute |
| `[10:20]` | `left_hand` | 10 | absolute |
| `[20:30]` | `right_hand` | 10 | absolute |
| `[30:33]` | `navigate_command` (`vx, vy, yaw_rate`) | 3 | absolute |
| `[33:34]` | `base_height_command` | 1 | absolute |

### 32 维是不是模型上限？

不是。OpenPI 的 `action_dim` 是可配置项，不存在“最多只能 32 维”的代码限制。需要区分两件事：

- 发布的 `pi05_base` checkpoint 是以 `action_dim=32` 训练的，因此它的动作输入/输出投影参数原生是 32 维。
- 本适配把模型实例设为 `action_dim=34`，并由 `ShapeAdaptedCheckpointWeightLoader` 只对三个动作投影数组执行
  受控的 32→34 扩展：已有 32 维权重原样继承，新增两维相关权重以零初始化并参与训练。其他参数若 shape
  不匹配仍会直接报错，避免静默加载错误 checkpoint。

30 维 state 会由 π0.5 的模型变换补到配置的 34 维；34 维 action 不截断、不补零。这里扩展的是模型投影，
不是伪造数据维度。

当前 pickup 数据的末四维在全部样本中恰好恒定为 `[0, 0, 0, 0.76]`。保留它们是为了与 RoboJuDo-plus
统一接口；但只用这份数据训练的 checkpoint 没见过移动或高度变化，不能期待它可靠地产生非零导航速度或
其他高度。要学习这些能力，后续必须加入相应命令发生变化、且视觉与状态覆盖充分的 episode，并重新计算
normalization stats 后继续训练。

## 1. 进入仓库并设置路径

下面所有命令均在 OpenPI 仓库执行：

```bash
cd /home/breeze/Desktop/workplace/openpi

export HF_LEROBOT_HOME=/home/breeze/Desktop/workplace/Isaac-GR00T/demo_data
export OPENPI_DATA_HOME=/home/breeze/.cache/openpi
export G1_REPO_ID=g1_23dof_upper_body_pickup_mulcam_openpi
```

`HF_LEROBOT_HOME` 必须指向数据集的父目录，不能直接指向数据集目录。配置中的 LeRobot repo ID 是数据集
目录名。

如需安装或更新依赖：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

## 2. 验证已修复的数据集

检查所有元数据声明的视频是否从零时间戳开始：

```bash
uv run scripts/fix_lerobot_video_timestamps.py \
  "$HF_LEROBOT_HOME/$G1_REPO_ID" \
  --check-only
```

预期摘要：

```text
Video start-time distribution:
  0.000000s: 333
Outside +/-0.0001s tolerance: 0
```

以后遇到同类新数据，可创建独立的修复副本：

```bash
uv run scripts/fix_lerobot_video_timestamps.py \
  /path/to/source_dataset \
  --output-dir /path/to/source_dataset_openpi
```

脚本不会覆盖源数据或已经存在的输出目录。

## 3. 选择训练配置

建议先使用：

```text
pi05_g1_pickup_lora
```

该配置同时为 PaliGemma 主干和 action expert 训练 LoRA 参数，action horizon 为 16 帧，即
`16 / 30 ≈ 0.53` 秒；batch size 为 8，训练 10,000 steps。按照仓库已有的 LoRA 训练方式，该配置关闭
EMA。

显存足够且确实需要全量微调时，可使用：

```text
pi05_g1_pickup
```

两个配置使用相同的数据变换和 π0.5 base checkpoint。batch size 必须能被 JAX 设备数量整除；若显存不足，
可在 `src/openpi/training/config.py` 中降低 batch size。

## 4. 运行单 batch 数据管线测试

该测试不需要 normalization stats。这里特意使用 `num_workers=0`，因为脚本从 stdin 启动；正式训练配置
使用四个 worker。

```bash
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python - <<'PY'
import dataclasses

from openpi.training import config
from openpi.training import data_loader

cfg = dataclasses.replace(
    config.get_config("pi05_g1_pickup_lora"),
    num_workers=0,
)
loader = data_loader.create_data_loader(
    cfg,
    num_batches=1,
    skip_norm_stats=True,
)
observation, actions = next(iter(loader))

print("state:", observation.state.shape, observation.state.dtype)
print("actions:", actions.shape, actions.dtype)
print("images:", {key: value.shape for key, value in observation.images.items()})
print("tokens:", observation.tokenized_prompt.shape)
PY
```

batch size 为 8 时，预期 shape：

```text
state:   (8, 34)
actions: (8, 16, 34)
images:  三个 (8, 224, 224, 3) tensor
tokens:  (8, 200)
```

上面展示的是模型变换后的 batch，原始 state 仍是 30 维。该步骤会同时验证：视频解码、task prompt 提取、
字段映射、三相机转换、34 维 action chunk、仅双臂 delta 转换、tokenization 和 state 的 30→34 补零。

## 5. 计算 normalization stats

开始训练前，针对所选配置计算统计值：

```bash
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
uv run scripts/compute_norm_stats.py \
  --config-name pi05_g1_pickup_lora
```

LoRA 配置的输出位置是：

```text
assets/pi05_g1_pickup_lora/g1_23dof_upper_body_pickup_mulcam_openpi/norm_stats.json
```

不要直接使用数据集自带的 `meta/stats.json`。OpenPI 必须在“仅左右臂 absolute-to-delta、其余 24 维保持
absolute”的变换之后，对完整 34 维 action 重新统计。如果选择 `pi05_g1_pickup`，需要把命令中的 config
name 换成该名字，因为每个训练配置有独立的 assets 目录。

训练前应检查生成的文件。当前数据的 action `[30:34]` 没有变化，因此它们的 `q01`/`q99` 极窄是已知且
符合预期的；其他本应活动的 state/action 维度若也出现极窄区间，则需要检查采集或字段映射。OpenPI 的
quantile normalization 在分母加入 `1e-6`，不会因这四个常量通道除零，但这些通道仍不提供变化监督。

## 6. 开始训练

推荐的首次训练命令：

```bash
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py \
  pi05_g1_pickup_lora \
  --exp-name=g1_pickup_v1 \
  --overwrite
```

checkpoint 保存到：

```text
checkpoints/pi05_g1_pickup_lora/g1_pickup_v1/<step>
```

配置每 1,000 steps 保存一次。不要默认最后一个 checkpoint 最好；建议首次对比 2,000、5,000、8,000 和
10,000 step 的实际机器人成功率。

恢复中断的训练时，使用相同实验名和 `--resume`：

```bash
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py \
  pi05_g1_pickup_lora \
  --exp-name=g1_pickup_v1 \
  --resume
```

不能同时传入 `--resume` 和 `--overwrite`。

## 7. 启动推理服务

例如加载 10,000 step：

```bash
OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_g1_pickup_lora \
  --policy.dir=checkpoints/pi05_g1_pickup_lora/g1_pickup_v1/10000
```

机器人运行时必须发送经过训练侧 repack 后的 G1 接口：

```python
observation = {
    "observation/head_rgb": head_rgb_uint8_hwc,
    "observation/left_wrist_rgb": left_wrist_rgb_uint8_hwc,
    "observation/right_wrist_rgb": right_wrist_rgb_uint8_hwc,
    "observation/state": state_float32_30,
    "prompt": "pick up the red cup",
}
```

图像物理视角和方向必须与训练数据一致。策略响应为：

```text
actions: (16, 34) float32 complete RoboJuDo targets
```

如需送入与 GR00T 相同的分组式控制接口，可以直接使用：

```python
from openpi.policies import g1_policy

groups = g1_policy.split_action_groups(response["actions"])
# groups["left_arm"]             (16, 5)
# groups["right_arm"]            (16, 5)
# groups["left_hand"]            (16, 10)
# groups["right_hand"]           (16, 10)
# groups["navigate_command"]     (16, 3): vx, vy, yaw_rate
# groups["base_height_command"]  (16, 1): height
```

闭环控制初期建议每次只执行前 4～8 个动作，然后重新请求策略。上真机前必须在机器人侧实现关节位置限位、
速度限位、单步变化限位、通信 watchdog 和急停。对于仅由当前静态底盘数据训练的 checkpoint，部署层还应
默认把末四维限制在安全静止值，直到它通过包含移动数据的仿真和真机验证。

## 数据集特别说明

- 当前数据有两个任务：`pick up the red cup` 和 `pick up the the bag and put it into the tray`。第二个任务
  有重复的 `the`。建议在计算 stats 和训练前修正 `meta/tasks.jsonl`，同时保证其他镜像任务元数据一致；不
  修正也不会阻止加载。
- 当前接口按 RoboJuDo 配置，只将左臂 5 维和右臂 5 维解释为 relative 训练目标。如果未来数据的双臂 action
  已经是 delta，应在 `LeRobotG1DataConfig` 中设置 `use_relative_arm_actions=False`，并重新计算 stats。
- 34 维顺序固定为：左臂 5、右臂 5、左手 10、右手 10、导航 3、高度 1。
- 当前数据保留末四维，但因为它们没有变化，只能监督“静止、0.76 高度”；RoboJuDo-plus 的移动能力需要
  含变化导航和高度命令的数据。
