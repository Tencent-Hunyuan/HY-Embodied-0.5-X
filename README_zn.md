# HY-Embodied-0.5-X

### An Enhanced Embodied Foundation Model for Real-World Agents

*Tencent Robotics X × HY Vision Team*

<p align="center">
  <a href="https://tairos.tencent.com/openSourceModels/hy-embodied"><img src="https://img.shields.io/badge/%E9%A1%B9%E7%9B%AE-%E4%B8%BB%E9%A1%B5-blue" alt="项目主页"></a>
  <a href="https://huggingface.co/tencent/HY-Embodied-0.5-X"><img src="https://img.shields.io/badge/Models-HuggingFace-yellow?logo=huggingface&logoColor=white" alt="HuggingFace"></a>
  <a href="https://github.com/Tencent-Hunyuan/HY-Embodied-0.5-X"><img src="https://img.shields.io/badge/GitHub-Repo-181717?logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://tairos.tencent.com/"><img src="https://img.shields.io/badge/TAIROS-%E5%B9%B3%E5%8F%B0-brightgreen" alt="TAIROS"></a>
  <a href="./README.md"><img src="https://img.shields.io/badge/README-English-red" alt="English README"></a>
  <a href="./docs"><img src="https://img.shields.io/badge/%E6%96%87%E6%A1%A3-Reference-6A5ACD?logo=readthedocs&logoColor=white" alt="文档"></a>
</p>

---

**HY-Embodied-0.5-X** 是腾讯 Robotics X 联合混元团队发布并开源的面向具身任务落地的多模态大模型。模型基于 `HY-Embodied-0.5 MoT-2B` 架构打造（总参数 4B，仅激活 2B），特别针对真实机器人的核心闭环 —— **"感知、推理、决策"** —— 进行优化。

模型在 **10 个主流具身任务规划 benchmark** 上达到业界领先水平，其中 **7 个 benchmark 排名第一**（限于边缘侧领域模型）。相比通用多模态模型，HY-Embodied-0.5-X 更聚焦机器人在真实交互中的核心问题，重点增强了 **精细操作理解、空间推理、动作预测、风险判断、多模态指代理解和长程规划** —— 将模型从 *"看"* 推进到 *"做"*。

## 🔥 Updates

- **`[2026-04-24]`** 🚀 发布 **HY-Embodied-0.5-X**，在 HY-Embodied-0.5 MoT-2B 基础上针对具身任务做定向增强，并开源配套推理与训练代码。

## ⭐️ Key Features

1. 🧠 **更强的空间理解能力** —— 准确理解物体位置、场景布局、相对空间关系与操作状态，为动作决策提供可靠的感知基础。
2. 🔗 **更强的长程规划能力** —— 处理多步骤、强依赖的复杂任务，在连续交互中稳定完成任务拆解、动作规划与执行决策。
3. 🤖 **更强的具身交互能力** —— 具备任务解析、指代消解、动作决策、风险判断和失败反思能力，贴近真实机器人交互闭环。
4. 📦 **端侧友好** —— 基于 MoT-2B 架构，总参数 4B / 激活 2B，支持端侧部署与实时响应。

## 📖 模型亮点

### 一、丰富可靠的数据组成

HY-Embodied-0.5-X 融合了 **自采机器人第一视角操作数据、机械臂操作数据以及开源具身数据**，构建了覆盖操作理解、第一人称任务推理、多模态交互指代的高质量数据库：

- **机械臂 / 人手操作轨迹**：围绕状态理解、下一步动作预测、操作风险判断、失败诊断和候选动作优劣比较等任务进行专项构建。
- **第一视角具身任务**：覆盖细粒度动作识别、子任务进度判断、手部空间定位、深度估计、相对空间关系推理、相机位姿推断等多类能力。
- **多模态交互指代理解**：围绕 "把这个放到那里" 这类真实协作场景中的模糊指令，结合语音与手势构建训练数据。

所有核心数据均附带 **思维链（CoT）标注**，配套 "生成—校验—修正—评测反跑验证" 的完整数据质量闭环。同时将具身、互联网及 3D 数据纳入统一的标准化重建管道，使异质数据源转化为一致、高质的具身推理数据。

### 二、验证—扩展—全量的分阶段训练策略

训练上，模型采用 **"验证—扩展—全量"** 的分阶段迭代策略：

1. 先通过精选小规模高质量数据快速验证训练配置与数据清洗效果；
2. 逐步扩大训练规模；
3. 在确认最优数据组合与训练策略后启动全量训练。

该策略既提升训练效率，也尽可能确保每一分算力都投入到最有价值的数据上。

## 📊 Evaluation

### 综合评测结果

在覆盖规划、空间推理、具身问答、视觉指代与轨迹理解等方向的 **10 个开源 benchmark** 上，HY-Embodied-0.5-X 保持第一梯队表现。

<p align="center">
  <img src="./assets/Results-All-benchmarks.png" width="90%" />
</p>

### 与同尺寸开源模型对比

相比同尺寸开源模型，HY-Embodied-0.5-X 在规划、空间理解和具身交互等核心能力上展现出均衡优势：

<p align="center">
  <img src="./assets/Results-Comparison_with_Open-Source_Models.png" width="90%" />
</p>

### AI2Thor 具身规划基准

我们自建了基于 AI2Thor 仿真环境的具身规划基准，共包含 **1011 道任务**，覆盖厨房、卧室、客厅、浴室四大家居场景，考察导航、抓取、放置、开关操作和食物切割等能力。HY-Embodied-0.5-X 在长程操作、自我意识和空间理解上展现出明显优势：

<p align="center">
  <img src="./assets/Results-Planning-benchmark.png" width="90%" />
</p>

### PlaygroundX 仿真接入

HY-Embodied-0.5-X 在基于 Tairos 平台的 **PlaygroundX 仿真架构** 上完成接入验证，可在 "把土豆扔到垃圾桶里""关上冰箱门""把西红柿放进冰箱" 等典型家居任务中生成完整规划，并根据环境反馈动态调整执行策略 —— 形成完整的 **ReAct 闭环**：感知 → 规划 → 执行 → 失败检测 → 重规划。

## 🛠️ 环境安装

仓库提供一键 conda 环境配置脚本 `setup_env.sh`，会自动完成 Python 3.12 环境创建、PyTorch / flash_attn / transformers（原生支持 HY-Embodied 的版本）及其他依赖的安装。`flash_attn` 需要从源码编译，耗时约 10–20 分钟：

```bash
bash setup_env.sh
conda activate hy_embodied_x

# （可选）把本项目作为 Python 包安装，获得 hy-embodied-train / hy-embodied-infer 命令
pip install -e .
```

### Prerequisites

| 项目    | 要求                         |
|---------|------------------------------|
| OS      | Linux                        |
| Python  | 3.12                         |
| CUDA    | 12.6                         |
| PyTorch | 2.10.0                       |
| GPU     | NVIDIA GPU with ≥ 16 GB VRAM |

> 核心依赖：`transformers`（[指定 commit](https://github.com/huggingface/transformers/commit/9293856c419762ebf98fbe2bd9440f9ce7069f1a)，原生支持 HY-Embodied）、`flash_attn==2.8.3`、`accelerate`、`deepspeed`、`timm`、`liger-kernel`。
> 完整依赖清单见 `setup_env.sh` 和 `requirements.txt`。

## 📥 下载权重

```bash
hf download tencent/HY-Embodied-0.5-X \
    --local-dir ckpts/HY-Embodied-0.5-X
```

权重（`*.safetensors`）已在 `.gitignore` 中忽略，默认放在 `ckpts/HY-Embodied-0.5-X/` 下。推理/训练代码也直接支持 HF Hub 仓库 id，会按需下载。

## 🚀 Quick Start

### 单图推理

```bash
# 默认：thinking 模式关闭
python -m hy_embodied.cli.infer \
    --model ckpts/HY-Embodied-0.5-X \
    --image ./assets/demo.jpg \
    --prompt "描述这张图片"

# 启用 thinking 模式（思维链推理）
python -m hy_embodied.cli.infer \
    --model ckpts/HY-Embodied-0.5-X \
    --image ./assets/demo.jpg \
    --prompt "描述这张图片" \
    --enable-thinking
```

旧入口 `python inference.py ...` 也保留，等价于上面的命令。

### Python API 示例

```python
import torch
from hy_embodied.inference import GenerationConfig, HyEmbodiedPipeline

pipe = HyEmbodiedPipeline.from_pretrained(
    "ckpts/HY-Embodied-0.5-X",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

# 默认：thinking 关闭
print(pipe.generate(
    "详细描述这张图片。",
    image="./assets/demo.jpg",
    generation_config=GenerationConfig(max_new_tokens=32768, temperature=0.05),
))

# 启用 thinking 模式
print(pipe.generate(
    "详细描述这张图片。",
    image="./assets/demo.jpg",
    generation_config=GenerationConfig(
        max_new_tokens=32768,
        temperature=0.05,
        enable_thinking=True,
    ),
))
```

更多批量 / 多图 / 视频用法参见 [`docs/inference.md`](./docs/inference.md)。

### OpenAI 兼容 API 服务

HY-Embodied-0.5-X 内置了 OpenAI 兼容的 API 服务，暴露标准 `/v1/chat/completions` 接口，可直接使用 OpenAI Python SDK、`curl` 或任何兼容客户端调用：

```bash
# 方式一：一键启动脚本（推荐）
bash scripts/run_server.sh

# 方式二：Python 模块
python -m hy_embodied.cli.server \
    --model ckpts/HY-Embodied-0.5-X \
    --host 0.0.0.0 --port 8080

# 方式三：console script（需先 pip install -e ".[serve]"）
hy-embodied-server --model ckpts/HY-Embodied-0.5-X --port 8080
```

然后即可使用标准 OpenAI SDK 调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")

# 纯文本（默认 thinking 关闭）
resp = client.chat.completions.create(
    model="HY-Embodied-0.5-X",
    messages=[{"role": "user", "content": "如何打开冰箱？"}],
)
print(resp.choices[0].message.content)

# 带图片（默认 thinking 关闭）
resp = client.chat.completions.create(
    model="HY-Embodied-0.5-X",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            {"type": "text", "text": "描述这张图片。"},
        ],
    }],
)

# 启用 thinking 模式（思维链推理）
resp = client.chat.completions.create(
    model="HY-Embodied-0.5-X",
    messages=[{"role": "user", "content": "如何打开冰箱？"}],
    extra_body={"enable_thinking": True},
)

# 流式输出
stream = client.chat.completions.create(
    model="HY-Embodied-0.5-X",
    messages=[{"role": "user", "content": "规划清理桌面的步骤。"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

也支持 `curl` 调用：

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HY-Embodied-0.5-X",
    "messages": [{"role":"user","content":"你好！"}]
  }'

# 启用 thinking 模式
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HY-Embodied-0.5-X",
    "messages": [{"role":"user","content":"你好！"}],
    "enable_thinking": true
  }'
```

服务启动后可访问 `/docs` 查看自动生成的 Swagger API 文档。完整服务端文档参见 [`docs/inference.md`](./docs/inference.md)。

### 坐标与响应格式

- **Point**：`(x, y)` 或 `[(x1, y1), (x2, y2)]`
- **Box**：`[xmin, ymin, xmax, ymax]`
- 坐标归一化为整数范围 **(0, 1000)**
- **Thinking 模式**（启用后）：响应结构为 `<think>[reasoning]</think><answer>[answer]</answer>`
- **直接模式**（默认）：响应仅包含答案，无推理部分

## 🔧 SFT 微调

```bash
# 单 GPU 快速烟测 — 无需 torchrun / DeepSpeed（显存 ≥ 16 GB 即可）
CUDA_VISIBLE_DEVICES=0 python -m hy_embodied.cli.train \
    --config configs/sft/example_small_single_gpu.yaml
# 或直接使用便捷脚本：
bash scripts/run_sft_single_gpu.sh

# 单机 8 卡（DeepSpeed ZeRO-2）
bash scripts/run_sft_1node_8gpu.sh

# 4 机 8 卡
bash scripts/run_sft_4node_8gpu.sh
```

仓库内提供两份参考配置：

- `configs/sft/example_small_single_gpu.yaml` — **单卡配置**，禁用 DeepSpeed，可直接 `python -m` 启动（无需 `torchrun`），适合快速验证和调试。
- `configs/sft/example_small.yaml` — **多卡配置**，默认开启 DeepSpeed ZeRO-2，需通过 `torchrun` 或 `accelerate` 启动。训练/优化器相关默认值即为发布训练所用的推荐值；一般情况下新用户**只需要修改 `data.train_data_paths` / `data.train_data_sampling_ratios`**，指向自己的 JSONL 数据组合即可。

两份配置均默认使用仓库自带的 `data_examples/data_demo.jsonl`（14 条样本，覆盖 6 个能力，图像已打包进仓库），因此上述命令可以零外部数据直接跑通。

数据格式、`/think` / `/no_think` 模式、训练/推理差异等详见 [`docs/training.md`](./docs/training.md) 与 [`docs/data_format.md`](./docs/data_format.md)。

## 📁 目录结构

```
HY-Embodied-0.5-X/
├── README.md / README_zn.md
├── LICENSE                   # Apache-2.0
├── pyproject.toml            # 打包 + 控制台命令
├── requirements.txt          # 完整依赖清单
├── setup_env.sh              # 一键安装脚本
│
├── src/hy_embodied/          # Python 包
│   ├── cli/                  # `python -m hy_embodied.cli.train / .infer / .server`
│   ├── training/             # SFT 训练器、数据管道、chat template
│   └── inference/            # HyEmbodiedPipeline
│
├── configs/
│   ├── sft/                  # 训练配置（example_small.yaml）
│   ├── accelerate/           # accelerate 启动器配置
│   ├── deepspeed/            # ZeRO 配置
│   └── fsdp/                 # FSDP 配置
│
├── scripts/                  # 单机 / 多机训练脚本 / API 服务启动
├── data_examples/            # 按能力类别的样例 JSONL（附 README）
├── docs/                     # data_format / training / inference / architecture
├── assets/                   # README / 演示用图
├── ckpts/                    # （gitignore）权重下载目录
├── outputs/                  # （gitignore）训练产物
└── inference.py              # 老 CLI 的向后兼容薄封装
```

详细的模块分层与依赖方向见 [`docs/architecture.md`](./docs/architecture.md)。

## 🎯 适用场景

HY-Embodied-0.5-X 适用于以下具身智能场景：

- **家庭服务 / 桌面操作**：真实环境下的空间推理、精细操作推理、任务理解与失败反思。
- **任务规划与仿真评测**：仿真环境中的规划评测与多模态交互研究。
- **本地部署与开发**：端侧具身能力验证与二次开发。

## 📚 Citation

```bibtex
@article{tencent2026hyembodied05x,
  title   = {HY-Embodied-0.5-X: An Enhanced Embodied Foundation Model for Real-World Agents},
  author  = {Tencent Robotics X and HY Vision Team},
  year    = {2026}
}
```

## 🙏 Acknowledgements

感谢 Hugging Face 社区和所有开源贡献者。HY-Embodied-0.5-X 的开源希望为具身智能社区提供一个更具落地导向的基座选择，推动模型从 "通用理解" 走向 "真实执行"。
