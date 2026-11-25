# Robot ROS Bridge

This folder connects the existing streaming camera/ASR/dialogue demo to the tabletop robot.

The current design deliberately uses a separate ROS2 process:

- `realtime_camera_asr_dialogue_demo.py` runs in the conda environment because it needs VLM/LLM/ASR Python dependencies.
- `robot_ros/reply_to_robot_bridge.py` runs with ROS2 Jazzy system Python because it needs `rclpy` and robot ROS messages.
- The current Haru2 simulator uses `/haru2/...` topics. The old `/idmind_tabletop/...` namespace is still supported through `--robot-namespace`.

## 1. Test Robot Commands

Run this in a ROS2 terminal:

```bash
source /opt/ros/jazzy/setup.bash
source /home/crystal/ros2_ws/install/setup.bash
cd /home/crystal/音乐/action_recognition/featurebased

PYTHONPATH=$PWD:$PYTHONPATH python3 -m robot_ros.robot_motion_test
```

If you only want to test motion:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m robot_ros.robot_motion_test --no-tts
```

If you only want to test TTS:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m robot_ros.robot_motion_test --no-motion --text "你好，我可以说话了。"
```

If you want to test English TTS emotion/prosody in the Haru2 Docker simulator:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m robot_ros.docker_tts_emotion_test
```

First check whether prosody control works at all:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m robot_ros.docker_tts_emotion_test \
  --emotion neutral,high_fast,low_slow \
  --no-emotion-tag
```

To hear only the built-in TTS emotion tags without manual prosody settings:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m robot_ros.docker_tts_emotion_test \
  --emotion happy,calm,sad,angry \
  --emotion-only
```

Test only one emotion:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m robot_ros.docker_tts_emotion_test --emotion happy
```

Test the same English sentence with several emotions:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m robot_ros.docker_tts_emotion_test \
  --emotion happy,calm,sad,angry \
  --text "I understand. Let us take this one step at a time."
```

If you are testing outside Docker with a normal ROS2 terminal, use `robot_ros.tts_emotion_test` instead.

## 2. Bridge Dialogue Replies To Robot

Keep the camera/ASR/dialogue demo running as before. Then open a new ROS2 terminal:

```bash
source /opt/ros/jazzy/setup.bash
source /home/crystal/ros2_ws/install/setup.bash
cd /home/crystal/音乐/action_recognition/featurebased

PYTHONPATH=$PWD:$PYTHONPATH python3 -m robot_ros.reply_to_robot_bridge \
  --dialogue-json streaming_camera_outputs/camera_asr_dialogue/dialogue_turns.json \
  --robot-namespace /haru2
```

Dry-run mode prints what would be sent without controlling the robot:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m robot_ros.reply_to_robot_bridge \
  --dialogue-json streaming_camera_outputs/camera_asr_dialogue/dialogue_turns.json \
  --dry-run
```

For the legacy namespace:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m robot_ros.reply_to_robot_bridge \
  --dialogue-json streaming_camera_outputs/camera_asr_dialogue/dialogue_turns.json \
  --robot-namespace /idmind_tabletop
```

## Notes

- The bridge watches `dialogue_turns.json` and sends each new robot reply to TTS.
- Before speaking, it performs a small acknowledgement motion.
- This keeps robot control independent from the VLM/ASR/LLM process, which avoids Python environment conflicts.
