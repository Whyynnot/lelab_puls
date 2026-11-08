# 开发与贡献

## 目录

```text
lelab/src/lelab/          LeLab 后端、训练、采集、原生模型任务
lelab/src/frontend/       React / TypeScript 页面
lelab/src/tests/          LeLab 核心测试
lelab/bridge/             原生会话、RTC 模式、USB/相机及 FP8 控制扩展
checkpoint.py            checkpoint 严格解析、名称兼容及分片写入
fp8_runtime.py            PI05 FP8 导出和真实 scaled GEMM
rtc_policy.py            FP8 服务的策略调用
serve.py                 仅监听 localhost 的模型 HTTP 服务
scripts/                 安装、构建、环境检查及样本准备
config/                  不含私人信息的示例配置
docs/                    教程、架构及故障排查
```

使用分离目录保留现有扩展的相对路径，降低迁移改动。普通 Load Models 的启动链为 `start_lelab.sh → bridge/launch.py → bridge/app.py → lelab.rollout → bridge/native_runner.py → 官方 lerobot_rollout`。

## 测试

在仓库根目录运行：

```bash
source .venv/bin/activate
python -m pytest tests lelab/bridge lelab/src/tests/test_training_config.py \
  lelab/src/tests/test_model_cache.py lelab/src/tests/test_rollout.py
cd lelab/src/frontend
npm test
npx tsc --noEmit -p tsconfig.app.json
npm run build
```

如果未安装系统 Node，可使用 `nodejs-wheel-binaries` 中的 Node，或在根目录运行 `bash scripts/build_frontend.sh`。更大范围的上游测试可运行 `python -m pytest lelab/src/tests`；部分旧测试依赖原上游接口，遇到版本差异请在 Issue 中报告。

默认排除 `local_artifact` 和 `gpu` 测试，测试使用模拟总线、合成图像、临时文件及隔离的子进程，不要求真实机械臂。GPU 测试需 SM89+：`python -m pytest test_fp8_runtime.py -m gpu`。真实产物测试请设置自己的 `LELAB_TEST_ARTIFACT` 和 `LELAB_TEST_MANIFEST`，运行 `python -m pytest test_checkpoint_compat.py lelab/bridge/test_fp8_integration.py -m local_artifact`，不要提交这些文件。

## 提交规范

Issue 请说明版本、复现步骤、预期/实际行为，附去除凭据、设备序列号和私有路径后的首个异常。PR 应有单一明确目的，并说明代码变化及验证；涉及动作执行、停止、队列或 USB 恢复时，加入能验证真实故障边界的测试。

不要提交 `.env`、token、缓存、权重、视频、采集数据、校准和运行日志。修改源自上游的文件时保留版权声明；新增贡献按 Apache-2.0 许可发布。上游可通用的修复欢迎拆分后向 [LeLab](https://github.com/huggingface/leLab) 或 [LeRobot](https://github.com/huggingface/lerobot) 提交。
