# Linux / WSL 硬件配置和排错

## 串口和相机

先确认系统能看到设备：

```bash
ls -l /dev/serial/by-id/
ls -l /dev/v4l/by-id/
```

串口访问通常需要 `dialout`，相机需要 `video`。按系统权限要求配置组成员，之后重新登录；不要通过 `chmod 777` 长期开放所有设备。常见 Ubuntu 设置：

```bash
sudo usermod -aG dialout,video "$USER"
```

WSL 用户还需在 Windows 使用 usbipd-win，将设备附加到所用 WSL 发行版。参考 [usbipd-win 上游教程](https://github.com/dorssel/usbipd-win/wiki/WSL-support)。该步骤在 Windows 执行；请按自己安装的版本使用对应语法。本仓库不会默认执行管理员脚本或自动附加未知设备。

## 可选：稳定身份映射

```bash
cp config/hardware.example.json config/hardware.local.json
```

将示例值替换为自己系统 `by-id` 中的完整路径。可以只配置部分角色，例如只配置 `follower`；没有配置的角色继续使用页面中的选择。

然后在 `.env` 写入真实绝对路径：

```bash
LELAB_HARDWARE_CONFIG=/home/YOUR_USER/lelab_puls/config/hardware.local.json
```

重启服务。设备存在时，扩展会更新已保存 SO-101 记录的路由，保留校准文件名、相机旋转和已有参数。缺失的相机不会被另一台相机代替；所需的已配置设备不可访问时，启动请求会被拦截。`hardware.local.json` 被 Git 忽略。

`GET /api/deploy/hardware-status` 提供配置的 USB 是否存在、是否可访问、启动检查及空闲恢复状态。**USB 可访问不代表电机供电正常，也不代表相机正在出帧。**

## 可选：自己的 Windows 恢复脚本

只有设置 `LELAB_USB_RECOVERY_SCRIPT` 才会调用 Windows helper，例如 `C:\Users\YOUR_USER\usb_connect.ps1`。脚本须接受 `-Action Start -AllowMissing -AsJson`，stdout 输出 JSON，可含 `Issues: []` 和 `Devices: []`。stderr 可输出诊断信息；非零退出码表示失败。此接口用于复用已有、经自己验证的设备附加流程，本仓库不提供与私人 USB 编号绑定的脚本。

空闲检查可尝试此 helper；正在采集、标定、遥操作或推理时不自动恢复 USB。运行中的 USB 掉线应先停止并处理物理连接。helper 超时会终止调用，但不能保证它创建的 Windows 后台子进程也退出，请让自己的脚本保持有界执行。

## 相机识别和预览

Linux/WSL 用后端 `/api/deploy/cameras` 枚举 V4L2 设备，并通过独立进程生成 MJPEG 预览。浏览器不必访问被转发到 WSL 的相机。界面中的 `wsl-camera:*` 是内部标识，Linux 也使用这个协议。

优先使用 `/dev/v4l/by-id/*-video-index0`，避免相机重插后 `/dev/video0`、`/dev/video2` 交换角色。没有 by-id 的设备回退到 video 索引，需要用户重插后重新核对。推理/采集时预览会暂停，避免两个进程争抢相机。

默认 `640×480 / 30 FPS / MJPG` 适合原开发环境的 UVC 相机；其他相机须选用自己支持的格式。输入图像旋转、相机语义和 checkpoint 特征名必须与训练一致。

### 室内灯光频闪

先用 `v4l2-ctl --device=/dev/v4l/by-id/YOUR_CAMERA --list-ctrls-menus` 查询相机实际支持的控制项（Ubuntu 可安装 `v4l-utils`）。不要把另一台相机的参数名或枚举值直接套用。

在 50 Hz 照明环境下，可先设置 `power_line_frequency=1`。如果已是 50 Hz 仍有明暗闪烁，可尝试固定 20 ms 曝光并关闭动态帧率；运动模糊较明显时再比较 10 ms。这是需要现场观察的候选设置，不能保证消除所有 LED 调光/PWM 条纹。

以下命令只适用于菜单与控制名匹配的相机，执行前停止录制/推理并保存原参数：

```bash
CAMERA=/dev/v4l/by-id/YOUR_CAMERA
v4l2-ctl --device="$CAMERA" --get-ctrl=power_line_frequency,auto_exposure,exposure_time_absolute,exposure_dynamic_framerate
v4l2-ctl --device="$CAMERA" --set-ctrl=power_line_frequency=1,exposure_dynamic_framerate=0
v4l2-ctl --device="$CAMERA" --set-ctrl=auto_exposure=1
v4l2-ctl --device="$CAMERA" --set-ctrl=exposure_time_absolute=200
```

这里 `auto_exposure=1` 须在该相机菜单中表示 Manual；曝光单位为 100 µs，因此 200 表示 20 ms、100 表示 10 ms，见 [Linux V4L2 相机控制文档](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html)。改变曝光会影响亮度与运动模糊，采集前应检查是否匹配预期。控制值通常在打开/关闭采集设备时保留，但 USB 断电重插可能恢复默认，重连后请回读确认。

## 常见故障

| 现象 | 含义及处理 |
| --- | --- |
| `Missing motor IDs ... found {}` | 串口可开，但电机总线没有回应；检查独立电源、串口选择、控制板至首个电机总线连接。重插电机总线前断电，机械臂先放稳 |
| `OpenCVCamera(...).read thread is not running` | 相机读线程已退出；查看更早的取帧/USB 错误，停止任务，再重新附加或插接并单独验证预览 |
| Windows“端口重置失败”、多个设备同时消失 | 优先排查共用 Hub、电源、线材和 USB 转发；仅重启页面不能修复物理链路 |
| 推理失败后 `Failed to disable torque` | 可能是清理过程的第二个错误；查首个 `LELAB_INFERENCE_FAULT` 和日志时间顺序，不应只针对最后一行排查 |
| 停止后仍显示正在释放设备 | 进程可能尚未退出或阻塞在 USB 内核调用。扩展会报告未停止；不要启动第二个任务抢占串口 |
| 采集未到目标条数就停止 | 检查失败的 episode 和相机最新帧时间；先验证已完成 episode 的元数据、数据帧和视频完整性，再上传或继续采集 |
| Hub 上传遇到 Xet 内部 I/O 异常 | 排除本地磁盘和权限后，可在 `.env` 设置 `HF_HUB_DISABLE_XET=1` 再重试；这是可选排障措施，不是所有上传失败的通用修复 |

软件保护不能保证设备断电/断连时机械臂保持姿态。做首次验证、重插和停止排错前请给机械臂支撑；不要让自动重启成为运动循环。
