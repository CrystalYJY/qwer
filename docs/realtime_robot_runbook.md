# Realtime Robot Demo Runbook

这个文档按当前实体/仿真测试流程写。你已经把 Haru2 仿真和摄像头相关容器打开后，优先使用“一键启动”脚本。

前置条件：

- Haru2 仿真已经启动。
- Azure Kinect / 摄像头容器已经启动，ROS2 能读到摄像头话题。
- 当前麦克风设备是 `plughw:2,0`。如果 `arecord -l` 变化了，要同步改 `runtime_configs/realtime_robot_dialogue.json` 里的 `mic_device`。
- 当前 ASR 默认使用讯飞“实时语音转写大模型”：`asr_engine=rtasr_llm`。如果新服务页的 AppID/APIKey/APISecret 和旧语音听写不同，要同步改 `xunfei_config.py`。
- U 盘目录 `/media/crystal/KINGSTON/qwer` 已经挂载。每次实验数据会直接写到这个目录下面。

## 一键启动

```bash
cd /home/crystal/音乐/action_recognition/featurebased

./scripts/run_realtime_robot_demo.sh
```

这个脚本会同时启动：

- 摄像头 frame bridge：从摄像头话题保存实时帧。
- 主程序：VLM 动作理解、ASR、对话 LLM、主动说话策略。
- Haru2 persistent bridge：把机器人回复发给 TTS/动作，并写入 TTS 状态。
- 日志监听：自动 `tail -f dialogue.log`。

启动后终端会先打印本次实验目录，例如：

```text
[launcher] experiment dir: /media/crystal/KINGSTON/qwer/robot_experiments/20260721_165536_entity_test
[launcher] output dir: /media/crystal/KINGSTON/qwer/robot_experiments/20260721_165536_entity_test/camera_asr_dialogue
[launcher] frame dir: /media/crystal/KINGSTON/qwer/robot_experiments/20260721_165536_entity_test/frame_bridge
```

按 `Ctrl-C` 会一起停止摄像头 frame bridge、主程序、机器人 bridge 和日志监听，并写入本次实验的 `metadata.json` 和总索引。

## 实验数据保存位置

所有本次实验数据都直接保存在：

```text
/media/crystal/KINGSTON/qwer/robot_experiments/<时间>_<实验标签>/
```

目录结构：

```text
<实验目录>/
  camera_asr_dialogue/
    dialogue.log
    vision.log
    system.log
    dialogue_turns.json
    skeleton_motion.json
    tts_status.json
    vlm_frame_evidence.json
    keyframes/
    trigger_frames/
  frame_bridge/
    latest.json
    frame_*.jpg
  code_state/
  metadata.json
  summary.md
```

总索引文件：

```text
/media/crystal/KINGSTON/qwer/robot_experiments/experiment_index.csv
/media/crystal/KINGSTON/qwer/robot_experiments/experiment_index.jsonl
```

其中：

- `dialogue_turns.json`：只记录真正发生过的用户-机器人对话或机器人主动说话；每条里有 `llm_visual_context`，列出本轮 LLM 看到的 `vlm-n`、完整描述、VLM 上传触发原因和对应图片。
- `vlm_frame_evidence.json`：完整视觉流水；记录每次 VLM 上传实际用了哪几张图、是什么触发上传、VLM 返回了什么。
- `keyframes/`：真正提交给 VLM 的关键帧；传了 3 张就保存 3 张，传了 4 张就保存 4 张。
- `trigger_frames/`：动作变快/动作触发调度时的那一帧截图。
- `metadata.json`：保存本次实验摘要和 `runtime_configs/realtime_robot_dialogue.json` 的配置快照。

## 常用启动变体

给实验换标签：

```bash
EXPERIMENT_LABEL=phone_test ./scripts/run_realtime_robot_demo.sh
```

指定本次实验目录名：

```bash
RUN_ID=pilot_01 ./scripts/run_realtime_robot_demo.sh
```

如果你已经单独启动了摄像头 frame bridge，不想让一键脚本再启动一个：

```bash
START_CAMERA_BRIDGE=0 FRAME_DIR=/path/to/existing/frame_bridge ./scripts/run_realtime_robot_demo.sh
```

如果摄像头话题名不同：

```bash
CAMERA_IMAGE_TOPIC=/your/camera/topic ./scripts/run_realtime_robot_demo.sh
```

如果 Haru2 容器名不同：

```bash
HARU2_CONTAINER=your-container-name ./scripts/run_realtime_robot_demo.sh
```

如果 skeleton 话题不同：

```bash
SKELETON_TOPIC=/your/skeleton/topic ./scripts/run_realtime_robot_demo.sh
```

## 运行中看什么

正常启动后应该陆续看到：

```text
CAMERA_READY
VLM_SUBMIT
VLM_RETURN
ASR_SPEECH
USER_ASR
MEMORY_READ
ROBOT
TTS_STATUS
```

机器人主动说话时会看到：

```text
PROACTIVE_SELECTED
PROACTIVE_TRIGGER
PROACTIVE_ROBOT
```

机器人 TTS 播放时，主程序会暂停 ASR 触发，避免机器人把自己的声音当成用户说话：

```text
TTS_STATUS ... playing=True
ASR_SUPPRESS
```

如果只看到 `ASR_READY`，说明主程序没有识别到麦克风输入，优先检查 `mic_device`。

## 当前机器人动作策略

当前默认配置是 `dialogue_output_mode=english`，LLM 直接输出英文短句给机器人 TTS。

LLM 每次回复会同时给出 `response_style`：`neutral`、`positive`、`care`、`sad`、`angry`。如果风格是 `neutral`，机器人走普通 TTS 和原来的轻微点头/说话动作；如果是非中性风格，bridge 会使用系统自带 emotion TTS，并调用对应 routine 表情：`positive -> harumoji_happy`、`care -> sympathetic`、`sad -> dejected`、`angry -> harumoji_angry`。

- 空闲时：根据骨骼追踪持续看向用户。
- 检测到 `ASR_SPEECH` 或 `USER_ASR`：轻微前倾/侧头，表示正在听。
- 普通对话回复：先做一个短暂思考动作，再轻点头眨眼，随后播放英文 TTS。
- 主动开口：先轻微转向/抬头并眨眼，再播放英文 TTS。
- TTS 播放时：做 1 到 3 次轻微点头/眼神变化，让它不像音箱。

如果你觉得“听到用户说话就动”太频繁，可以给 bridge 加：

```bash
--no-listen-motion
```

如果你想保留点头/听话动作，但不要说话时的轻微节奏动作，可以加：

```bash
--no-speaking-motion
```


如果头部追踪方向反了，需要临时修改 `scripts/run_realtime_robot_demo.sh` 里 bridge 启动参数，或单独启动 bridge 时加：

```bash
--track-yaw-scale -4.0
```

如果动作不明显，试：

```bash
--track-yaw-scale 6.0 --track-pitch-scale 3.0
```

当前主程序还启用了自适应视觉调度：

```text
画面/骨架稳定：按基础间隔提交 VLM，默认约 12 秒一次
动作密集：关键点速度或连续帧差超过阈值时，立即触发一次额外 VLM 提交
事件保护：动作触发最小间隔默认约 3 秒，避免 API 请求过密
并发保护：base VLM 普通请求最多 `vlm_max_inflight` 个同时在路上，priority VLM 强动作请求最多 `vlm_priority_max_inflight` 个同时在路上
强动作优先：如果动作分数达到 priority 阈值，会进入独立 priority worker，日志显示 `VLM_PRIORITY_PENDING` 和 `schedule=motion_event_priority`
连续强动作：priority worker 忙时不堆积多个请求，只保留最新强动作；如果等待超过 `vlm_priority_pending_max_age_s` 会丢弃
```

其中关键点运动来自 Haru2 bridge 已经订阅的 skeleton 话题，主要看 `head/nose/hand/wrist/elbow` 等关键点的速度；图像帧差作为兜底信号。

当前 skeleton 触发不是直接用单帧速度阈值，而是先经过 `meaningful_action_trigger`：

```text
最近约 3 秒 skeleton 运动
→ 提取峰值速度、持续运动比例、活跃关节数量、手/头运动等特征
→ 计算 meaningful action score
→ 分数超过阈值才触发额外 VLM
```

这个模块目前是规则评分版，接口已经按小模型方式设计。后续可以把评分函数替换成 RandomForest、XGBoost、MLP 或 TCN，而不需要改主程序的数据流。

对应配置在：

```text
runtime_configs/realtime_robot_dialogue.json
```

关键参数是 `adaptive_visual_schedule`、`adaptive_min_event_interval_s`、`meaningful_action_trigger`、`meaningful_action_score_threshold`、`meaningful_action_min_peak_speed`、`adaptive_motion_threshold`、`vlm_max_inflight`、`vlm_priority_max_inflight`、`vlm_priority_motion_threshold` 和 `vlm_priority_pending_max_age_s`。

主动说话分成两路：动作评论和沉默兜底。动作评论更优先，但动作评论说出后会锁住后续两轮用户对话，避免机器人一直抓动作插话。

```text
salient_action_comment：
VLM 刚返回一条有互动价值的明显动作观察时立即判断，围绕这个动作说一句。

general_checkin：
如果用户短时间不说话，机器人结合最近视觉上下文低打扰搭话，但不做强动作评论。
```

VLM 不维护固定动作表，因为动作是开放集。它每次返回视觉描述时，只额外返回最小互动字段：

```text
{
  "observation": "用户坐在办公椅上，正在进行手臂伸展运动。",
  "confidence": 0.9,
  "salience": 3,
  "comment_style": "care"
}
```

字段含义：

```text
observation：一句自然语言动作观察，作为对话锚点
confidence：VLM 对观察的置信度
salience：互动价值评分，不是动作类别
  0 = 静止/日常状态：坐着、看屏幕、安静阅读、手放着
  1 = 常规小动作：打字、鼠标、轻微转头、小幅手部动作、随手拿物体
  2 = 明显可评论动作：喝水、伸展、站起、专门看手机、翻找东西、拿文件或工具
  3 = 强互动/强变化：看向机器人、挥手、明显求助、走动、大幅伸展、突然离开
comment_style：回复语气，neutral / question / care / playful / offer_help / encouragement
```

`salient_action_comment` 的触发逻辑：

```text
VLM 返回 confirmed 观察
→ confidence 通过视觉记忆阈值
→ salience >= proactive_action_salience_threshold，默认 2
→ 用户短暂停顿 >= proactive_visual_event_quiet_s，默认 2 秒
→ action_lock_user_turns_remaining == 0
→ 没有待处理 ASR，用户没有正在说话
→ 不是刚评论过的同一个视觉事件
→ 使用 salient_action_comment prompt
→ 说出后锁住接下来 proactive_action_lock_user_turns 轮用户对话，默认 2 轮
```

`general_checkin` 的触发逻辑：

```text
定时 proactive check
→ 距离最近一次用户说话或机器人说话 >= proactive_quiet_s，默认 20 秒
→ 没有待处理 ASR，用户没有正在说话
→ 使用 general_checkin prompt
→ 可以轻量参考 primary_visual_event，但避免“今天怎么样”这类泛泛问候
```

触发方式：

```text
VLM 刚返回时只尝试 salient_action_comment
定时检查只尝试 general_checkin
同一个视觉事件只评论一次，避免对同一个动作重复说话
动作评论后，两轮用户对话内不再因为动作主动开口
动作锁期间如果用户沉默很久，只能由 general_checkin 触发
general_checkin 之间至少间隔 proactive_quiet_s，避免沉默期间反复搭话
机器人回复正在生成、等待 TTS 播放或 TTS 播放中时，不会再启动新的主动回复
如果 proactive 生成期间用户开始说话或 ASR 已有结果，这句 proactive 会被丢弃
如果动作不值得评论，就不走动作评论
```

两个 prompt 的目标：

```text
salient_action_comment：
必须围绕 primary_visual_event，说一句自然短句；不要说“我看到你……”；不要像监控；不要泛泛 check-in。

general_checkin：
用户安静了一段时间；结合 primary_visual_event 说一句低打扰搭话；不要过度评价动作，不要泛泛问候。
```

为了避免话太多，主动说话仍受这些限制：

```text
用户正在说话时不说
TTS 播放时不说
机器人上一句回复正在生成或等待播放时不说
同一个视觉事件不重复说
动作评论后，默认 2 轮用户对话内不再动作触发
```

## 机器人桥接快速测试或排错

功能：不经过 ASR/LLM，直接测试仿真机器人能不能动、能不能说话。

测试动作：

```bash
cd /home/crystal/音乐/action_recognition/featurebased

python3 -m robot_ros.docker_haru2_bridge \
  --container haru-simulator-web-server-1 \
  --test-motion
```

测试 TTS：

```bash
cd /home/crystal/音乐/action_recognition/featurebased

python3 -m robot_ros.docker_haru2_bridge \
  --container haru-simulator-web-server-1 \
  --test-tts "你好，我准备好了。"
```

只打印将要执行的 Docker/ROS 命令，不真的控制机器人：

```bash
python3 -m robot_ros.docker_haru2_bridge \
  --container haru-simulator-web-server-1 \
  --test-motion \
  --dry-run
```



说明：`docker_haru2_persistent_bridge` 是低延迟版本。它只在启动时 `docker exec` 一次，并在容器内常驻 ROS publisher；旧的 `docker_haru2_bridge` 每条动作都会重新 `docker exec + ros2 topic pub --once`，延迟更高，只建议排错时使用。

## 推荐检查顺序

1. 一键启动终端出现 `CAMERA_READY`，说明摄像头 frame bridge 正常。
2. 出现 `VLM_SUBMIT` 和 `VLM_RETURN`，说明视觉链路正常。
3. 出现 `USER_ASR`，说明麦克风和讯飞 ASR 正常。
4. 出现 `ROBOT`，说明对话 LLM 正常。
5. 出现 `new dialogue reply` 或 `new proactive reply`，说明机器人 bridge 读到回复。
6. 仿真机器人说话/轻点头，说明 Docker 内 ROS2 控制链路正常。

## 常见问题

### 只有 VLM，没有 USER_ASR

检查麦克风设备：

```bash
arecord -l
```

如果设备号从 `plughw:2,0` 变了，改 `runtime_configs/realtime_robot_dialogue.json` 的 `mic_device`。

当前 ASR 推荐参数已经写在 `runtime_configs/realtime_robot_dialogue.json`：

```text
asr_engine = rtasr_llm
asr_energy_threshold = 300
asr_min_speech_frames = 3
asr_max_silence_frames = 9
asr_min_recognition_s = 1.2
asr_final_wait_s = 2.5
```

`USER_ASR_BUFFER` 表示系统已经识别到一句话，但会短暂等待用户是否继续补充。当前等待时间是：

```text
dialogue_turn_delay_s = 1.4
```

如果机器人还是经常在你没说完时回复，把 `runtime_configs/realtime_robot_dialogue.json` 里的 `dialogue_turn_delay_s` 调到 `1.8`。如果觉得反应慢，可以调回 `1.0`。

如果要临时退回旧版语音听写接口，把配置改成：

```json
"asr_engine": "iat"
```

如果又出现很多 `ASR_EMPTY`，优先把 `asr_energy_threshold` 调高到 `350` 或 `400`。如果小声说话经常检测不到，再降到 `250`。

如果日志是反复出现：

```text
ASR_SPEECH
ASR_CONNECTED
ASR_EMPTY
```

说明麦克风检测到了声音，但讯飞没有返回文字。当前 ASR 客户端已经做了“连接讯飞时继续缓存音频”的处理；如果仍然频繁空识别，优先排查：

- 麦克风离人太远或指向不对。
- 机器人 TTS / 电机动作 / 键盘声触发了 ASR。
- 环境声太大，能量够触发但语音不清晰。

如果确认是机器人 TTS 回灌，先检查本次实验目录里的 `camera_asr_dialogue/tts_status.json`。播放时应有 `"playing": true`，并且 `dialogue.log` 里应出现 `TTS_STATUS` 和 `ASR_SUPPRESS`。如果没有，确认一键启动脚本里的 bridge 命令包含 `--tts-status-json`。

排查时可以临时打开电平：

```bash
/home/crystal/miniconda3/envs/st_tr_emotion/bin/python -m streaming_robot_action.realtime_camera_asr_dialogue_demo \
  --config-json runtime_configs/realtime_robot_dialogue.json \
  --asr-level-meter \
  --console-mode all
```

如果想验证是不是机器人“听话动作”干扰麦克风，可以在 bridge 命令里临时加：

```bash
--no-listen-motion
```

### keyframes 调试图

本次实验目录里的 `camera_asr_dialogue/keyframes` 是 VLM 上传前保存的关键帧截图，主要用于检查“到底给模型看了什么”。不影响记忆和对话逻辑。

现在每次实验都会创建新的实验目录，一般不需要手动清理 keyframes。

### VLM 经常 timeout

当前配置已经比原始命令更稳：

```text
vlm_interval_s = 12
keyframes_normal = 3
vlm_timeout_s = 15
hard_timeout_s = 18
```

如果仍然超时，可以继续增大 `vlm_interval_s` 到 `15`。

### 机器人说旧话

新版 bridge 默认跳过第一份已有 `dialogue_turns.json`。如果还出现旧话，先停止 bridge 后重新启动即可。
