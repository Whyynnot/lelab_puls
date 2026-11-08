# 可选：PI05 FP8 部署

这条入口用于已经训练好的、SO-101 六维 state/action、两路真实相机的 PI05 checkpoint。普通 Load Models 的 SmolVLA RTC 不依赖本功能。

## 量化范围

`fp8_runtime.py` 导出 `pi05-fp8-w8a8-v1`：语言 Transformer 的 q/k/v/o/gate/up/down 线性层使用 E4M3 FP8。权重采用每矩阵 absmax 缩放，激活采用每次调用动态 absmax 缩放；推理执行 `torch._scaled_mm`，输出 BF16。视觉、action expert、embedding、norm 等按原生浮点类型保留；没有 INT4 或伪量化回退。

CPU 可以导出；实际推理要求 CUDA SM89+。当前不支持带 visual/proprioceptive memory 的模型。不将基座大维度输出截取成六维；相机特征、动作顺序、归一化统计必须与训练 checkpoint 一致。

## 1. 下载原始 checkpoint

先进入虚拟环境，并设置模型缓存环境。以下都是需要替换的占位符：

```bash
source .venv/bin/activate
export LELAB_CACHE_ROOT=/absolute/path/to/lelab_cache
export HF_HOME="$LELAB_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub-live"
hf download OWNER/MODEL --include 'checkpoints/004000/pretrained_model/*'
```

使用下载命令返回的真实本地 checkpoint 路径，确认其中有 `config.json`、原始模型权重、`policy_preprocessor.json`、`policy_postprocessor.json` 及相应统计/资源文件。子目录和 step 的补零规则取决于具体仓库，不固定为 `004000`。模型许可和访问权限由模型发布者决定。

## 2. 导出到单独目录

```bash
.venv/bin/python fp8_runtime.py \
  --source /absolute/path/to/pretrained_model \
  --output /absolute/path/to/my_pi05_fp8 \
  --device cpu --threads 2
```

源和输出必须分离；非空输出不会覆盖。完成前存在 `EXPORT_INCOMPLETE`，服务会拒绝加载。完成后应有 `quantization.json` 和分片索引。不要手动删除不完整标记伪装完成；重新导出到新目录。原始权重和 FP8 目录都可以放在已挂载的 E 盘。

## 3. 创建本地 profile

```bash
mkdir -p lelab/profiles
cp config/pi05-fp8.example.json lelab/profiles/my_pi05_fp8.json
```

填写绝对 `artifact` 路径和模型真实的两路图像特征名，`name` 与文件名一致。默认 `allow_robot=false`，可先进行离线验证。profile 的相机角色固定为 `head/wrist`，特征值可映射到训练时不同的图像键。

tokenizer 优先使用产物中的 `tokenizer/tokenizer.json`；否则沿用 checkpoint 保存的处理器。如果配置 `tokenizer_model`，必须是与该模型一致的 PaliGemma SentencePiece 文件，不能使用任意 tokenizer 代替。

## 4. 用真实保存观测离线测试

准备自己的 `raw-observation.npz`：`raw_state` 是按 profile 顺序排列的六维物理关节值；`head`、`wrist` 是训练方向一致的 RGB、uint8、HWC 图像。不要把 BGR 当成 RGB。然后执行：

```bash
.venv/bin/python scripts/prepare_fp8_sample.py \
  --profile lelab/profiles/my_pi05_fp8.json \
  --observation /absolute/path/to/raw-observation.npz \
  --task 'YOUR EXACT TRAINING TASK'
bash start_lelab.sh
```

打开 **http://localhost:8000/quantized**，选择 profile、加载模型，等服务就绪后运行离线测试。样本准备使用保存的处理器和统计，不会打开电机/相机；离线测试会执行真实模型前向，但不会发送机器人动作。输出形状/有限值正常只是必要检查，不能证明任务成功或运动安全。

## 5. 实机验证

确认模型、输入、反归一化、动作单位和离线结果后，在本地 profile 将 `allow_robot` 改为 `true`。刷新页面，填写自己的从臂端口、对应校准文件、真实相机 JSON 与任务提示词，先做短时验证。

运行中普通/RTC 切换由扩展控制循环管理。模型服务独占 GPU 模型，机器人进程独占电机 I/O；请先停止机器人，再卸载模型。该入口不与普通 Load Models 同时执行动作。停止状态显示未退出时，不要启动第二个任务。

还可以单独启动模型服务用于不连接机器人的 HTTP 调试：

```bash
.venv/bin/python serve.py --artifact /absolute/path/to/my_pi05_fp8 --steps 10
```

默认仅监听 `127.0.0.1:8765`。页面要求自行管理从其他终端启动的服务；不要把无认证的机器人控制接口直接开放到公网。
