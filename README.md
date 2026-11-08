# LeLab Puls

基于 [Hugging Face LeLab](https://github.com/huggingface/leLab) 的社区增强版，面向使用 SO-101、USB 相机和本地 VLA 模型的用户。目标是把实际采集、训练和推理中积累的改进整理成可复用、可讨论的代码。

[English quick start](docs/README.en.md) · [推理模式与流程图](docs/inference.md) · [硬件与故障排查](docs/hardware.md) · [PI05 FP8](docs/fp8.md) · [开发与贡献](CONTRIBUTING.md)

这是独立的社区分支，与上游 LeLab 保持 Apache-2.0 许可及来源声明。仓库名保留为 `lelab_puls`。

## 提供的改进

| 功能 | 使用入口 | 行为 |
| --- | --- | --- |
| LeRobot v0.7.0 适配 | 安装脚本 | 适配训练配置、校准、策略加载和原生 rollout 接口 |
| 可配置模型缓存 | `.env` | `LELAB_CACHE_ROOT` 统一模型、Hub、Torch 缓存，可放到 E 盘 |
| 训练配置导入与编辑 | Training | 导入 JSON/YAML、编辑完整配置、导出配置；选择预训练权重并传递原生训练参数 |
| Load Models 的 RTC | 推理配置窗口 | 为满足条件的 SmolVLA、PI0、PI05 提供自动/普通/RTC 选择，启动前校验兼容性 |
| 推理控制与可视化 | 推理页面 | 显示会话归属，暂停/重置/恢复、修改任务及支持的 chunk 参数、显示观测和预测 |
| 相机与串口稳定性 | 相机选择、可选设备映射 | Linux/WSL 后端预览、稳定 USB 身份、MJPG 默认值、独立预览进程和启动检查 |
| 故障及停止反馈 | 推理页面 | 优先展示原始观测异常，限制清理等待时间，进程未退出时不报告停止成功 |
| PI05 FP8 扩展 | `/quantized` | 可选的语言层 FP8 W8A8 导出及独立模型服务，支持普通/RTC 控制循环 |

不包含模型权重、采集数据、账号凭据、校准文件或原开发机器的设备配置。所有模型请自行取得授权并下载。

## 1. 环境要求

- 推荐 Ubuntu 24.04 / WSL2 Ubuntu，Python **3.12**、Git、FFmpeg。
- CUDA 推理参考组合：PyTorch **2.10.0**、Torchvision **0.25.0**、TorchCodec **0.10.0**、CUDA **12.6**；Transformers **5.22.2**，LeRobot **0.7.0**。
- 普通模型的显存要求取决于模型；PI05 FP8 扩展需要 NVIDIA **SM89 或更新架构**。RTC 不等于量化，也不会自动减少模型显存。
- 增强启动器及硬件扩展依赖 Linux 的进程、串口与 V4L2 接口。macOS、原生 Windows 的完整扩展尚未验证；WSL 浏览器可运行在 Windows。
- 项目处于社区实验阶段。自动测试不替代对各自机械臂、模型和 USB 拓扑的实机验证。

## 2. 克隆并安装

建议使用独立虚拟环境，保留已有 LeLab 环境作为回退。

```bash
git clone https://github.com/Whyynnot/lelab_puls.git
cd lelab_puls
sudo apt-get update
sudo apt-get install -y python3.12-venv python3-dev build-essential git ffmpeg \
  libglib2.0-0 libegl1 libusb-1.0-0-dev portaudio19-dev
bash scripts/setup.sh
```

脚本创建 `.venv`，安装参考 CUDA PyTorch、官方 LeRobot v0.7.0 和本仓库中的 LeLab，使用随 Python 包安装的 Node 构建前端，最后运行只读环境检查。首次安装需要访问 GitHub、PyPI、PyTorch 和 npm。

如果已在本仓库 `.venv` 安装匹配版本的 PyTorch，可以运行 `bash scripts/setup.sh --skip-torch`。其他 PyTorch/TorchCodec 组合需要自行验证；约束文件不会自动适配它们。

模型可能需要 Hugging Face 登录或同意访问条款：

```bash
source .venv/bin/activate
hf auth login
```

不要把 token 写进配置文件或提交到 Git。

## 3. 配置缓存和设备

```bash
cp .env.example .env
```

编辑 `.env`。默认模型缓存是 `~/.cache/lelab`，WSL 下使用 E 盘可设置：

```bash
LELAB_CACHE_ROOT=/mnt/e/lelab_cache
```

请先确认磁盘已挂载且可写。这个设置覆盖模型相关的 `HF_HOME`、`HF_HUB_CACHE`、`HF_DATASETS_CACHE`、`HF_XET_CACHE`、`TORCH_HOME`；Hub 模型缓存位于 `$LELAB_CACHE_ROOT/huggingface/hub-live`。它不会自动搬迁旧缓存。

已有认证默认从原 `HF_HOME/token` 复用，也可显式设置 `HF_TOKEN_PATH`。录制数据默认仍使用 `~/.cache/huggingface/lerobot`（可设置 `HF_LEROBOT_HOME`）；LeLab 的设备记录及校准位置仍沿用上游 `~/.cache/huggingface/lerobot`。**模型缓存变量不会搬迁校准文件**。

USB 路径可直接在 LeLab 页面选择。若需要重插后稳定映射 `leader/follower/head/wrist`，参考 [硬件配置教程](docs/hardware.md)；设备映射和 Windows 自动恢复均为可选项。

## 4. 启动与停止

```bash
bash start_lelab.sh
```

浏览器打开 **http://localhost:8000**。启动器会加载仓库根目录 `.env` 并注册增强接口。直接运行上游 `lelab` 命令只启动核心页面，不会启用本仓库的全部 bridge 功能。

前台运行时用 `Ctrl+C` 停止服务。也可以后台管理：

```bash
.venv/bin/python lelab/bridge/manage.py start
.venv/bin/python lelab/bridge/manage.py status
.venv/bin/python lelab/bridge/manage.py stop
```

日志位于 `lelab/logs/`，运行状态位于 `lelab/runtime/`，均不提交到 Git。`stop` 发出终止请求；请用 `status` 和日志确认退出，再拔除缓存磁盘。机械臂失联或 USB 内核阻塞时，软件无法保证电机保持姿态，先让机械臂处于有支撑的位置。

## 5. 首次使用及普通模型推理

1. 在首页选择自己的机器人、主从臂串口并完成校准。已有校准只应复用于同一台设备。
2. 选择主相机和手腕相机，检查预览、旋转方向与训练时一致。相机键名和数量必须匹配模型输入。
3. 在 **Load Models** 中输入模型 Hub ID 或所需 checkpoint 路径，等待配置和权重准备完成。部分模型仓库将权重放在 `checkpoints/004000/pretrained_model` 等子目录，实际名称请以模型仓库为准。
4. 打开推理设置，填写训练时的任务提示词、选择从臂及相机。对于兼容模型，可选择**自动 / 普通 / RTC**；不兼容的模式会说明原因。
5. 先用短时任务验证。首次加载、预热与稳定运行时延不同。开始后观察真实机械臂和页面状态；停止时等待进程释放，避免两个任务同时占用串口。

SmolVLA 的 RTC 使用 LeRobot 原生 guided RTC。当前允许列表为 SmolVLA、PI0、PI05，且要求单帧观测、有效 chunk、无不兼容的历史/记忆/时序集成配置。ACT 不提供 guided RTC。普通/RTC 的差异、`M/N/FPS`、切换限制和两张流程图见 [推理教程](docs/inference.md)。

量化 PI05 产物必须走 `/quantized` 页面，不能通过普通模型加载器直接加载。

## 6. 训练配置教程

在 **Training → Configuration** 的“训练 Config · 导入与编辑”卡片中：

1. 点击“加载配置文件”，导入 LeRobot 的 `train_config.json`、JSON/YAML 或本仓库 [ACT 示例](config/training-act.example.yaml)。
2. 修改数据集、步数、batch size、学习率、checkpoint 间隔等字段，点击**应用配置**；未应用的编辑会阻止提交。
3. “预训练权重”控件用于选择是否加载权重及 Hub ID/本地目录。这里控制策略初始化；具体策略的 backbone/VLM 开关也会在提交时检查。
4. 其他原生选项放入 `extra_config`，例如 `policy` 下的冻结层、训练子模块、scheduler、PEFT/LoRA 等。**字段名取决于策略及 LeRobot v0.7.0**，请使用该策略配置中存在的字段。`extra_config` 中的值优先于表单默认值。
5. 可以导出当前配置，提交后查看任务记录及启动日志；本地/远程任务沿用 LeLab 原有执行方式。

例如，对支持这些字段的 SmolVLA，可在完整配置中加入：

```json
{
  "dataset_repo_id": "YOUR_HF_ACCOUNT/YOUR_DATASET",
  "policy_type": "smolvla",
  "policy_load_pretrained": true,
  "policy_pretrained_path": "lerobot/smolvla_base",
  "steps": 20000,
  "batch_size": 8,
  "extra_config": {
    "policy": {
      "freeze_vision_encoder": true,
      "train_expert_only": true
    }
  }
}
```

提交前后端使用原生 LeRobot 配置解析器检查未知字段和类型；不支持的组合会报错，不会静默丢弃。选择从头训练不代表所有大模型都适合随机初始化；请依照所用模型的训练方案配置。

## 7. 排错、更新与贡献

```bash
.venv/bin/python scripts/doctor.py
```

检查导入、依赖、原生 RTC 接口、前端构建及实际模型缓存路径，不启动机器人动作。故障处理见 [hardware.md](docs/hardware.md)，FP8 导出教程见 [fp8.md](docs/fp8.md)。

更新前先停止会话和服务，再执行：

```bash
git pull --ff-only
bash scripts/setup.sh
```

欢迎提交 Issue/PR。请附上环境版本、模型类型、普通或 RTC 模式、相机配置和去除个人信息的首个异常日志；不要上传 token、私人数据、校准文件或完整模型。测试命令与代码结构见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 来源与许可

- 核心来自 [huggingface/leLab](https://github.com/huggingface/leLab)，本次整理基线为 `9b5afaee33baa1d773e499937564f3831d25de22`，保留上游版权和 Apache-2.0 许可。
- 机器人、策略和 guided RTC 使用官方 [LeRobot v0.7.0](https://github.com/huggingface/lerobot/tree/v0.7.0)。未将本机电机参数改动固化进依赖。
- 本仓库的整理和社区改进说明见 [NOTICE](NOTICE) 与 [CHANGELOG.md](CHANGELOG.md)。模型、数据及第三方资产遵循各自许可。
