# 实时机器人主动说话策略设计

本文档整理当前实时机器人系统中“主动说话”的完整设计。这里描述的是代码当前真实实现，而不是早期讨论中的草案。

## 1. 设计目标

主动说话的目标不是让机器人频繁打断用户，而是让它在合适时机自然地说一句话：

1. 当用户有明显动作时，机器人可以围绕动作轻量评论。
2. 当用户和机器人都安静了一段时间时，机器人可以结合视觉上下文轻量搭话。
3. 当用户正在说话、机器人正在播放 TTS、或者用户即将发言时，机器人不主动插话。

当前主动说话分为两类：

| 类型 | 触发来源 | 作用 |
| --- | --- | --- |
| `salient_action_comment` | VLM 返回新的明显动作后 | 围绕最新可见动作说一句 |
| `general_checkin` | 定时器检测到长时间安静后 | 结合视觉上下文轻量搭话 |

## 2. 整体流程

整体链路如下：

```text
摄像头帧
  -> 视觉 buffer
  -> 基础间隔或动作事件触发 VLM
  -> VLM 返回 observation / confidence / salience / comment_style
  -> 写入视觉事件记忆
  -> 主动策略判断是否可以说话
  -> 构造主动回复 prompt
  -> Dialogue LLM 生成一句话
  -> 再次确认用户没有开始说话
  -> 输出机器人回复并抑制 ASR 回听
```

## 3. VLM 视觉观察

VLM 不是输出固定动作表，而是对最近多帧 keyframes 做开放式观察。

### 3.1 VLM 输出字段

当前 VLM 只返回四个字段：

```json
{
  "observation": "",
  "confidence": 0.0,
  "salience": 0,
  "comment_style": "neutral"
}
```

字段含义：

| 字段 | 含义 | 用途 |
| --- | --- | --- |
| `observation` | 一句自然语言视觉描述 | 给对话和主动回复作为视觉上下文 |
| `confidence` | 视觉判断置信度 | 低置信度事件不会成为可靠记忆 |
| `salience` | 动作是否值得机器人主动评论，0 到 3 | 主要决定动作型主动说话是否触发 |
| `comment_style` | 如果要说话，建议使用的语气 | 影响回复语气，不直接决定是否触发 |

### 3.2 当前 VLM prompt

当前 VLM prompt 位于 `streaming_robot_action/vlm_action_client.py`，核心内容如下：

```text
You are the visual observer of a social robot.

The following frames are sampled from the robot's recent visual stream.
Describe what the user appears to be doing.

Focus on visible body motion, hand motion, posture, and object interaction.
Do not over-infer hidden intention.
Answer naturally and briefly in Chinese, using one short sentence.

Also judge interaction-worthiness for a friendly desktop robot.
Do not match a fixed action list; use general social judgment for unseen actions.
Use this strict salience scale:
0=routine/static: sitting, watching screen, reading quietly, hands resting.
1=minor routine motion: typing, mouse use, slight head turn, small hand movement, casually holding an object.
2=clearly comment-worthy action: drinking, stretching, standing up, deliberately checking a phone, searching, handling papers or tools.
3=strongly comment-worthy or socially directed action: looking at the robot, waving, obvious help-seeking, walking around, big stretch, sudden leaving.
If unsure between two levels, choose the lower level.
comment_style: neutral, question, care, playful, offer_help, or encouragement.

Return JSON only: {"observation":"","confidence":0.0,"salience":0,"comment_style":"neutral"}
```

### 3.3 salience 的意义

`salience` 不是动作类别，而是“这个动作是否值得社交机器人主动说一句”的强度。

| salience | 当前含义 | 例子 |
| --- | --- | --- |
| 0 | 常规或静态状态 | 坐着、看屏幕、安静阅读、手放着 |
| 1 | 轻微常规动作 | 打字、用鼠标、轻微转头、小幅手部动作、随手拿着物体 |
| 2 | 明显且适合轻量评论的动作 | 喝水、伸展、站起来、明显看手机、翻找、处理文件或工具 |
| 3 | 强烈明显或有社交指向的动作 | 看向机器人、挥手、明显求助、走动、大幅伸展、突然离开 |

注意：

- 当前策略中，动作型主动说话要求 `salience >= 2`。
- `comment_style=care` 只代表“如果说话，语气偏关心”，不代表一定触发。
- 如果 VLM 把某个状态判为 `salience=0`，它不会触发动作型主动说话。

## 4. 视觉事件记忆

VLM 返回后会被解析成一个视觉事件，并写入动作记忆。

视觉事件中与主动说话相关的字段包括：

| 字段 | 含义 |
| --- | --- |
| `event_id` | 当前视觉事件 ID，用于避免同一个事件重复评论 |
| `observation_short` | 简短视觉描述 |
| `confidence` | 置信度 |
| `salience` | 互动价值等级 |
| `comment_style` | 建议回复语气 |
| `received_at` | VLM 返回时间，用于计算视觉事件新鲜度 |

对话系统读取视觉上下文时，会形成：

```json
{
  "current_visual_observation": {},
  "recent_visual_events": []
}
```

其中：

- `current_visual_observation` 是最近一条可用视觉事件。
- `recent_visual_events` 是最近若干条视觉事件。
- 当前配置中通常最多使用最近 3 条视觉事件。

## 5. 主动说话入口

主动说话有两个入口。

### 5.1 VLM 返回后触发

当 VLM 返回一条确认事件后，系统立刻尝试一次主动判断：

```text
VLM_RETURN confirmed
  -> get_dialogue_context()
  -> try_proactive_reply(trigger_source="visual_event")
```

这个入口只负责动作型主动说话，也就是 `salient_action_comment`。

适合场景：

- 用户伸展手臂。
- 用户拿起杯子喝水。
- 用户举起手机查看。
- 用户站起来。
- 用户看向机器人。

### 5.2 定时器触发

系统每隔一小段时间检查一次是否已经安静足够久：

```text
timer
  -> get_dialogue_context()
  -> try_proactive_reply(trigger_source="timer")
```

这个入口只负责沉默型主动说话，也就是 `general_checkin`。

适合场景：

- 用户一直坐着看屏幕。
- 用户一直没有说话。
- 没有明显动作，但可以轻量搭话。

## 6. 主动说话前的安全门控

无论是哪一种触发来源，都会先经过统一 gate。

当前会阻止主动说话的情况包括：

| gate | 含义 |
| --- | --- |
| `robot_output_busy` | 机器人刚刚说完或 TTS 仍在播放附近，不能连续说 |
| `asr_active` | ASR 正在检测到用户说话 |
| `tts_playing` | TTS 正在播放 |
| `no_visual_observation` | 没有可用视觉观察 |
| `visual_too_old` | 当前视觉观察超过最大新鲜度 |
| `pending_user_turn` | 用户语音已经进入缓冲，等待合并成完整用户轮次 |

设计原则是：

```text
宁可少说，也不要抢用户的话。
```

## 7. 动作型主动说话：salient_action_comment

动作型主动说话由 `trigger_source="visual_event"` 触发。

### 7.1 触发条件

必须同时满足：

1. 用户已经安静至少 `proactive_visual_event_quiet_s` 秒。
2. 当前视觉事件有有效 `event_id`。
3. 当前视觉事件没有被主动评论过。
4. 当前视觉事件 `salience >= proactive_action_salience_threshold`。
5. 动作 lock 没有生效。
6. 通过 ASR/TTS/视觉新鲜度等安全 gate。

当前配置：

```json
"proactive_visual_event_quiet_s": 2,
"proactive_action_salience_threshold": 2
```

也就是：

```text
用户至少 2 秒没说话，并且 VLM 判断 salience >= 2，才可能动作触发。
```

### 7.2 分数计算

动作型候选的分数是：

```python
score = salience * 2 + min(visual_count, 3)
```

系统还有一个总阈值：

```python
trigger_threshold = 5
```

因此：

| salience | 视觉事件数量 | 分数 | 结果 |
| --- | --- | --- | --- |
| 1 | 3 | 5 | 不触发，因为 salience 低于动作阈值 2 |
| 2 | 1 | 5 | 可以触发 |
| 2 | 3 | 7 | 稳定触发 |
| 3 | 3 | 9 | 强触发 |

### 7.3 动作 lock

动作型主动说话成功后，系统会设置：

```text
action_lock_turns_remaining = proactive_action_lock_user_turns
```

当前配置：

```json
"proactive_action_lock_user_turns": 2
```

含义是：

```text
机器人因为动作主动说过一句后，接下来 2 轮用户对话内，不再因为动作主动插话。
```

这个设计是为了避免机器人看到每个动作都评论，变成“动作播报员”。

注意：

- 动作 lock 只限制动作型主动说话。
- 如果用户和机器人安静足够久，沉默型 `general_checkin` 仍然可以触发。

## 8. 沉默型主动说话：general_checkin

沉默型主动说话由 `trigger_source="timer"` 触发。

### 8.1 触发条件

必须同时满足：

1. 用户和机器人都安静至少 `proactive_quiet_s` 秒。
2. 最近视觉事件数量大于 0。
3. 距离上一次 `general_checkin` 至少过去 `proactive_quiet_s` 秒。
4. 通过 ASR/TTS/视觉新鲜度等安全 gate。

当前配置：

```json
"proactive_check_interval_s": 2,
"proactive_quiet_s": 20
```

含义是：

```text
系统每 2 秒检查一次；
如果用户和机器人都安静了 20 秒，就考虑轻量搭话。
```

### 8.2 interaction_quiet 的计算

沉默触发不是只看“用户多久没说话”，而是看：

```python
interaction_quiet = now - max(last_user_turn_time, last_robot_reply_time)
```

也就是：

```text
从用户最后一次说话或机器人最后一次说话中较晚的那个时间开始算。
```

这样可以避免机器人刚刚回复完，马上因为“用户没说话”继续主动说。

### 8.3 分数计算

沉默型候选的分数是：

```python
score = 3 + min(visual_count, 3)
```

总触发阈值仍然是：

```python
trigger_threshold = 5
```

因此：

| 视觉事件数量 | 分数 | 结果 |
| --- | --- | --- |
| 0 | 无候选 | 不触发 |
| 1 | 4 | 低于阈值，不触发 |
| 2 | 5 | 可以触发 |
| 3 | 6 | 稳定触发 |

这意味着刚启动时，如果视觉记忆太少，即使沉默时间够了，也可能不会马上说话。

## 9. 两类主动说话如何避免冲突

当前设计不是让两类主动说话同时竞争，而是按触发入口分开：

| 触发入口 | 只产生的主动类型 |
| --- | --- |
| `visual_event` | `salient_action_comment` |
| `timer` | `general_checkin` |

冲突避免机制包括：

1. TTS 正在播放时不主动说话。
2. ASR 正在收音时不主动说话。
3. 用户语音进入 pending buffer 时不主动说话。
4. 机器人刚说完时，由 `robot_output_busy_until` 阻止连续说。
5. 同一个视觉事件不会被动作型主动说话重复评论。
6. 动作型主动说话后，接下来 2 轮用户对话内不再动作触发。
7. `general_checkin` 之间至少间隔一个 `proactive_quiet_s`。

整体效果是：

```text
动作明显时优先围绕动作说；
没有明显动作且双方安静足够久时，才轻量 check-in；
用户一旦开始说话，主动说话立即让路。
```

## 10. 主动回复生成 prompt

Policy 只决定“该不该说”和“属于哪一类”，真正说什么由 Dialogue LLM 生成。

主动回复 prompt 会包含：

- 主动类型。
- 触发原因。
- 当前视觉事件 `primary_visual_event`。
- `comment_style`。
- 最近视觉事件。
- 最近几轮对话。

### 10.1 动作型回复 prompt

当主动类型是 `salient_action_comment` 时，任务约束是：

```text
Task: make a brief comment anchored to primary_visual_event.
The sentence must naturally respond to that visible action.
Do not say 'I see you...' or sound like surveillance.
Do not give a generic check-in.
Use exactly one short sentence, about 6-14 words.
Do not stack multiple supportive comments or ask multiple questions.
Use comment_style to choose the tone.
```

效果目标：

```text
Nice stretch. Feeling a bit better?
```

而不是：

```text
How are you doing today?
```

### 10.2 沉默型回复 prompt

当主动类型是 `general_checkin` 时，任务约束是：

```text
Task: the user has been quiet for a while.
Say one low-pressure, open-ended check-in lightly grounded in primary_visual_event.
It may mention the user's current visible context or object if useful.
Do not produce a generic greeting like 'How are you doing today?'.
Do not sound like surveillance or over-comment on the action.
Do not give advice unless the user asks.
Use exactly one short sentence, about 6-14 words.
```

效果目标：

```text
Still focused on the screen—want to talk through anything?
```

而不是：

```text
Hope you're doing well today!
```

## 11. 主动回复生成后的二次确认

即使 policy 已经决定可以说，并且 Dialogue LLM 已经生成好回复，系统在真正输出前还会再次检查：

```text
ASR 是否已经检测到用户说话？
是否已经有 pending_user_turn？
ASR queue 里是否有新文本？
```

如果有，系统会记录：

```text
PROACTIVE_ABORT
```

然后取消这次主动说话。

这个机制很重要，因为 Dialogue LLM 生成回复需要几秒钟。在这几秒钟里，用户可能已经开始说话了。

## 12. TTS 播放与 ASR 抑制

主动回复输出后，系统会估算机器人说话时长，并抑制 ASR：

```text
robot_speech_s = estimate_robot_speech_s(reply, asr_suppress_after_robot_s)
asr.suppress_for(robot_speech_s)
```

同时，系统会读取 TTS 状态文件：

```json
"tts_status_json": "streaming_camera_outputs/camera_asr_dialogue/tts_status.json",
"tts_status_max_age_s": 2
```

如果 TTS 状态为 playing：

1. `proactive_policy.state.tts_playing = true`
2. ASR 临时停止收音
3. 主动说话不会再次触发
4. `robot_output_busy_until` 延长一小段时间

相关配置：

```json
"asr_suppress_after_robot_s": 3.0,
"asr_tts_status_hold_s": 1.5
```

## 13. 当前配置摘要

当前主动说话相关配置如下：

```json
{
  "enable_proactive": true,
  "proactive_check_interval_s": 2,
  "proactive_quiet_s": 20,
  "proactive_visual_event_quiet_s": 2,
  "proactive_action_lock_user_turns": 2,
  "proactive_action_salience_threshold": 2
}
```

解释：

| 配置 | 当前值 | 含义 |
| --- | --- | --- |
| `enable_proactive` | `true` | 开启主动说话 |
| `proactive_check_interval_s` | `2` | 每 2 秒检查一次沉默触发 |
| `proactive_quiet_s` | `20` | 用户和机器人都安静 20 秒后考虑 check-in |
| `proactive_visual_event_quiet_s` | `2` | 用户至少安静 2 秒后才允许动作触发 |
| `proactive_action_lock_user_turns` | `2` | 动作触发后，接下来 2 轮用户对话内不再动作触发 |
| `proactive_action_salience_threshold` | `2` | 动作型主动说话要求 `salience >= 2` |

## 14. 当前策略的典型行为

### 14.1 用户伸展手臂

```text
VLM: 用户坐在办公椅上伸展手臂
salience=2 或 3
trigger_source=visual_event
proactive_type=salient_action_comment
```

机器人可能说：

```text
Nice stretch. Feeling a bit better?
```

### 14.2 用户一直看屏幕

```text
VLM: 用户坐在桌前看电脑屏幕
salience=0
trigger_source=timer
双方安静 >= 20 秒
proactive_type=general_checkin
```

机器人可能说：

```text
Still focused on the screen—want to talk through anything?
```

### 14.3 用户正在说话

```text
ASR_SPEECH
asr_active=True
```

机器人不会主动说话。

### 14.4 机器人正在 TTS

```text
tts_playing=True
```

机器人不会触发下一句主动回复，ASR 也会被临时抑制。

## 15. 当前已知边界

1. `salience` 是动作型主动说话的核心触发信号。
2. `comment_style` 只控制语气，不控制触发。
3. 如果 VLM 把某个用户状态判成 `salience=0`，它不会触发动作型主动说话。
4. 沉默触发依赖最近视觉事件数量；视觉事件太少时可能分数不够。
5. VLM 请求如果超时，相关动作就不会及时进入视觉记忆，主动说话也可能错过时机。
6. 当前没有主动说话总次数上限，长时间测试不会因为次数达到上限而停止主动发话。

## 16. 日志阅读方式

调试主动说话时，重点看这些日志：

| 日志 | 含义 |
| --- | --- |
| `VLM_SUBMIT` | 提交了一次 VLM 观察 |
| `VLM_RETURN` | VLM 返回视觉观察，包含 `salience` 和 `style` |
| `VLM_UPDATE vlm_error` | VLM 请求失败或超时 |
| `PROACTIVE_SELECTED` | policy 选中了某个主动类型 |
| `PROACTIVE_TRIGGER` | 记录主动触发原因 |
| `PROACTIVE_ROBOT` | 主动回复最终生成并输出 |
| `PROACTIVE_ABORT` | 生成后发现用户开始说话，因此取消主动回复 |
| `PROACTIVE_SKIP` | debug 模式下记录为什么没触发 |
| `TTS_STATUS` | 当前 TTS 播放状态 |
| `ASR_SPEECH` | ASR 检测到用户开始说话 |

常见判断：

- 如果 `VLM_RETURN` 里 `salience=0`，动作型主动说话不会触发。
- 如果有 `PROACTIVE_ABORT`，说明系统本来想说，但用户在生成期间开始说话，所以取消了。
- 如果有 `tts_playing` 或 `asr_active`，说明系统为了避免抢话或回听而跳过了主动说话。
- 如果频繁 `Request timed out`，说明动作可能没有及时进入视觉记忆，主动触发会变慢或错过。

