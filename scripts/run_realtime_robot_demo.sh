#!/usr/bin/env bash
set -euo pipefail

# Resolve from this script so this launcher can never silently run the sibling
# "featurebased" checkout instead of the current entity checkout.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/runtime_configs/realtime_robot_dialogue.json"
CONTAINER="${HARU2_CONTAINER:-haru-simulator-web-server-1}"
# The physical robot is reachable from this host's sourced ROS workspace.
# Docker remains available by explicitly setting HARU2_BRIDGE_MODE=docker.
HARU2_BRIDGE_MODE="${HARU2_BRIDGE_MODE:-local}"
HARU2_DOCKER_SETUP="${HARU2_DOCKER_SETUP:-source /opt/ros/jazzy/setup.bash}"
HARU2_LOCAL_SETUP="${HARU2_LOCAL_SETUP:-source /opt/ros/jazzy/setup.bash && source /home/crystal/ros2_ws/install/setup.bash}"
HARU2_SSH_SETUP="${HARU2_SSH_SETUP:-source /opt/ros/jazzy/setup.bash}"
HARU2_SSH_TARGET="${HARU2_SSH_TARGET:-haru-32.local}"
HARU2_ROBOT_NAMESPACE="${HARU2_ROBOT_NAMESPACE:-}"
HARU2_MESSAGE_PACKAGE="${HARU2_MESSAGE_PACKAGE:-}"
HARU2_ROS_STATIC_PEERS="${HARU2_ROS_STATIC_PEERS:-}"
HARU2_ROS_AUTOMATIC_DISCOVERY_RANGE="${HARU2_ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
SKIP_HARU2_TOPIC_CHECK="${SKIP_HARU2_TOPIC_CHECK:-0}"
SKELETON_TOPIC="${SKELETON_TOPIC:-/strawberry/azure/skeletons}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/media/crystal/KINGSTON/qwer/robot_experiments}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-entity_test}"
SAVE_EXPERIMENT="${SAVE_EXPERIMENT:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${EXPERIMENT_LABEL}}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$EXPERIMENT_ROOT/$RUN_ID}"
OUTPUT_DIR="$EXPERIMENT_DIR/camera_asr_dialogue"
FRAME_DIR="${FRAME_DIR:-$EXPERIMENT_DIR/frame_bridge}"
START_CAMERA_BRIDGE="${START_CAMERA_BRIDGE:-1}"
REQUIRE_CAMERA_READY="${REQUIRE_CAMERA_READY:-1}"
CAMERA_READY_TIMEOUT_S="${CAMERA_READY_TIMEOUT_S:-12}"
CAMERA_IMAGE_TOPIC="${CAMERA_IMAGE_TOPIC:-/azure_kinect/rgb/image_raw/compressed}"
# Camera drivers normally publish sensor-data BEST_EFFORT. This subscription
# also remains compatible when the publisher is RELIABLE.
CAMERA_QOS="${CAMERA_QOS:-sensor_data}"
# This checkout consumes images through frame_dir.  Keeping the ROS subscriber
# in a dedicated process isolates camera callbacks from VLM/ASR/robot work.
CAMERA_FRAME_MODE="${CAMERA_FRAME_MODE:-latest_only}"
DIALOGUE_LOG="$OUTPUT_DIR/dialogue.log"
DIALOGUE_JSON="$OUTPUT_DIR/dialogue_turns.json"
SKELETON_MOTION_JSON="$OUTPUT_DIR/skeleton_motion.json"
VISUAL_FOCUS_JSON="$OUTPUT_DIR/visual_focus_trigger.json"
SHOW_CHINESE_SUBTITLES="${SHOW_CHINESE_SUBTITLES:-1}"
CHINESE_SUBTITLE_FONT_SIZE="${CHINESE_SUBTITLE_FONT_SIZE:-28}"

MAIN_PY="/home/crystal/miniconda3/envs/st_tr_emotion/bin/python"

MAIN_PID=""
BRIDGE_PID=""
CAMERA_PID=""
GPU_PID=""
TAIL_PID=""
SUBTITLE_PID=""
CLEANED_UP=0
SAVED_RUN=0

save_experiment() {
  if [[ "$SAVE_EXPERIMENT" != "1" || "$SAVED_RUN" == "1" ]]; then return; fi
  SAVED_RUN=1
  if [[ ! -d "$EXPERIMENT_ROOT" ]]; then
    echo "[launcher] experiment index skipped: destination not found: $EXPERIMENT_ROOT"
    return
  fi
  echo "[launcher] writing experiment metadata/index for $EXPERIMENT_DIR ..."
  "$MAIN_PY" "$ROOT/scripts/save_realtime_robot_experiment.py" \
    --dest "$EXPERIMENT_ROOT" \
    --experiment-dir "$EXPERIMENT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --frame-bridge-dir "$FRAME_DIR" \
    --config-json "$CONFIG" \
    --label "$EXPERIMENT_LABEL" || true
}

cleanup() {
  if [[ "$CLEANED_UP" == "1" ]]; then return; fi
  CLEANED_UP=1
  echo ""
  echo "[launcher] stopping realtime robot demo..."
  if [[ -n "${TAIL_PID}" ]]; then kill "${TAIL_PID}" 2>/dev/null || true; fi
  if [[ -n "${SUBTITLE_PID}" ]]; then kill "${SUBTITLE_PID}" 2>/dev/null || true; fi
  if [[ -n "${MAIN_PID}" ]]; then kill "${MAIN_PID}" 2>/dev/null || true; fi
  if [[ -n "${BRIDGE_PID}" ]]; then kill "${BRIDGE_PID}" 2>/dev/null || true; fi
  if [[ -n "${CAMERA_PID}" ]]; then kill "${CAMERA_PID}" 2>/dev/null || true; fi
  if [[ -n "${GPU_PID}" ]]; then kill "${GPU_PID}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  sleep 0.2
  save_experiment
}
trap cleanup EXIT INT TERM

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

echo "[launcher] experiment dir: $EXPERIMENT_DIR"
echo "[launcher] output dir: $OUTPUT_DIR"
echo "[launcher] frame dir: $FRAME_DIR"
echo "[launcher] Haru2 bridge mode: $HARU2_BRIDGE_MODE"
echo "[launcher] camera mode: $CAMERA_FRAME_MODE"

wait_for_camera_frames() {
  if [[ "$REQUIRE_CAMERA_READY" != "1" ]]; then return; fi
  CAMERA_READY_DEADLINE=$((SECONDS + CAMERA_READY_TIMEOUT_S))
  echo "[launcher] waiting for camera frames (${CAMERA_READY_TIMEOUT_S}s timeout)..."
  while (( SECONDS < CAMERA_READY_DEADLINE )); do
    if [[ -f "$FRAME_DIR/latest.json" ]] || compgen -G "$FRAME_DIR/frame_*.jpg" >/dev/null; then
      echo "[launcher] camera frames ready."
      return
    fi
    sleep 0.5
  done
  echo "[launcher] camera frames not ready; VLM would have no visual input." >&2
  echo "[launcher] Check camera Docker/driver and topic: $CAMERA_IMAGE_TOPIC" >&2
  bash -lc "source /opt/ros/jazzy/setup.bash && source /home/crystal/ros2_ws/install/setup.bash && ros2 topic list -t | grep -E 'azure_kinect|rgb|image_raw|compressed|camera' || true" >&2
  exit 1
}

if [[ "$CAMERA_FRAME_MODE" != "frame_bridge" && "$CAMERA_FRAME_MODE" != "latest_only" ]]; then
  echo "[launcher] CAMERA_FRAME_MODE='$CAMERA_FRAME_MODE' is unsupported in this checkout; use frame_bridge." >&2
  exit 2
fi

if [[ "$START_CAMERA_BRIDGE" == "1" ]]; then
  echo "[launcher] starting camera frame bridge..."
  CAMERA_FRAME_ARGS=""
  if [[ "$CAMERA_FRAME_MODE" == "latest_only" ]]; then CAMERA_FRAME_ARGS="--latest-only"; fi
  bash -lc "source /opt/ros/jazzy/setup.bash && source /home/crystal/ros2_ws/install/setup.bash && cd '$ROOT' && PYTHONPATH=\$PWD:\$PYTHONPATH python3 -m streaming_robot_action.ros_camera_frame_bridge --image-topic '$CAMERA_IMAGE_TOPIC' --qos '$CAMERA_QOS' --output-dir '$FRAME_DIR' --timeline-jsonl '$OUTPUT_DIR/camera_timeline.jsonl' $CAMERA_FRAME_ARGS" &
  CAMERA_PID=$!
  sleep 0.8
else
  echo "[launcher] camera frame bridge not started; expecting frames in $FRAME_DIR"
fi

wait_for_camera_frames

if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while true; do
      date '+[%F %T]'
      nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>&1
      sleep 2
    done
  ) >> "$EXPERIMENT_DIR/gpu.log" &
  GPU_PID=$!
  echo "[launcher] GPU monitor: $EXPERIMENT_DIR/gpu.log"
else
  echo "[launcher] nvidia-smi unavailable; GPU monitor not started."
fi

echo "[launcher] starting main VLM/ASR/dialogue process..."
"$MAIN_PY" -m streaming_robot_action.realtime_camera_asr_dialogue_demo \
  --config-json "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --frame-dir "$FRAME_DIR" \
  --skeleton-motion-json "$SKELETON_MOTION_JSON" \
  --visual-focus-json "$VISUAL_FOCUS_JSON" \
  --tts-status-json "$OUTPUT_DIR/tts_status.json" &
MAIN_PID=$!

if [[ "$SHOW_CHINESE_SUBTITLES" == "1" ]]; then
  echo "[launcher] starting Chinese translation subtitle window..."
  python3 -m streaming_robot_action.chinese_subtitle_qt \
    --dialogue-json "$DIALOGUE_JSON" --font-size "$CHINESE_SUBTITLE_FONT_SIZE" &
  SUBTITLE_PID=$!
fi

case "$HARU2_BRIDGE_MODE" in
  docker)
    BRIDGE_NAMESPACE="${HARU2_ROBOT_NAMESPACE:-/haru2}"
    BRIDGE_MESSAGE_PACKAGE="${HARU2_MESSAGE_PACKAGE:-haru2_core_msgs}"
    echo "[launcher] starting Docker Haru2 bridge with built-in head tracking..."
    echo "[launcher] Haru2 bridge namespace: $BRIDGE_NAMESPACE package: $BRIDGE_MESSAGE_PACKAGE"
    bash -lc "source /opt/ros/jazzy/setup.bash && source /home/crystal/ros2_ws/install/setup.bash && cd '$ROOT' && PYTHONPATH=\$PWD:\$PYTHONPATH python3 -m robot_ros.docker_haru2_persistent_bridge --container '$CONTAINER' --robot-namespace '$BRIDGE_NAMESPACE' --message-package '$BRIDGE_MESSAGE_PACKAGE' --setup '$HARU2_DOCKER_SETUP' --dialogue-json '$DIALOGUE_JSON' --dialogue-log '$DIALOGUE_LOG' --track-skeleton --skeleton-topic '$SKELETON_TOPIC' --track-interval-s 0.35 --skeleton-motion-json '$SKELETON_MOTION_JSON'" &
    BRIDGE_PID=$!
    ;;
  local)
    BRIDGE_NAMESPACE="${HARU2_ROBOT_NAMESPACE:-/haru2}"
    BRIDGE_MESSAGE_PACKAGE="${HARU2_MESSAGE_PACKAGE:-haru2_core_msgs}"
    BRIDGE_SETUP="$HARU2_LOCAL_SETUP"
    if [[ -n "$HARU2_ROS_STATIC_PEERS" ]]; then
      BRIDGE_SETUP="$BRIDGE_SETUP && export ROS_AUTOMATIC_DISCOVERY_RANGE='$HARU2_ROS_AUTOMATIC_DISCOVERY_RANGE' && export ROS_STATIC_PEERS='$HARU2_ROS_STATIC_PEERS'"
    fi
    echo "[launcher] starting direct local ROS publisher (same /haru2 topics as 2.py)..."
    echo "[launcher] robot namespace: $BRIDGE_NAMESPACE"
    if [[ -n "$HARU2_ROS_STATIC_PEERS" ]]; then
      echo "[launcher] Haru2 bridge static peers: $HARU2_ROS_STATIC_PEERS"
    fi
    if [[ "$SKIP_HARU2_TOPIC_CHECK" != "1" ]]; then
      if ! bash -lc "source /opt/ros/jazzy/setup.bash && source /home/crystal/ros2_ws/install/setup.bash && ros2 topic list -t | grep -E '^${BRIDGE_NAMESPACE}/|tts|TTS|cmd_tts|routine' >/dev/null"; then
        echo "[launcher] local Haru2 topics/services are not visible in this ROS2 environment." >&2
        echo "[launcher] Continuing anyway, because some entity setups only appear after publishers start." >&2
        echo "[launcher] If the robot stays silent, check ROS_DOMAIN_ID, ROS_IP, robot network, or HARU2_ROBOT_NAMESPACE." >&2
        echo "[launcher] Current namespace: $BRIDGE_NAMESPACE; set SKIP_HARU2_TOPIC_CHECK=1 to hide this warning." >&2
      fi
    fi
    bash -lc "$BRIDGE_SETUP && cd '$ROOT' && PYTHONPATH=\$PWD:\$PYTHONPATH python3 -m robot_ros.reply_to_robot_bridge --dialogue-json '$DIALOGUE_JSON' --robot-namespace '$BRIDGE_NAMESPACE' --tts-status-json '$OUTPUT_DIR/tts_status.json' --track-skeleton --skeleton-topic '$SKELETON_TOPIC' --track-interval-s 0.10 --track-max-age-s 5.0 --skeleton-motion-json '$SKELETON_MOTION_JSON' --visual-focus-json '$VISUAL_FOCUS_JSON' --timeline-jsonl '$OUTPUT_DIR/robot_timeline.jsonl'" &
    BRIDGE_PID=$!
    ;;
  ssh)
    BRIDGE_NAMESPACE="${HARU2_ROBOT_NAMESPACE:-/haru2}"
    BRIDGE_MESSAGE_PACKAGE="${HARU2_MESSAGE_PACKAGE:-haru2_core_msgs}"
    echo "[launcher] starting SSH Haru2 bridge on $HARU2_SSH_TARGET with built-in head tracking..."
    echo "[launcher] Haru2 bridge namespace: $BRIDGE_NAMESPACE package: $BRIDGE_MESSAGE_PACKAGE"
    bash -lc "source /opt/ros/jazzy/setup.bash && source /home/crystal/ros2_ws/install/setup.bash && cd '$ROOT' && PYTHONPATH=\$PWD:\$PYTHONPATH python3 -m robot_ros.docker_haru2_persistent_bridge --ssh-target '$HARU2_SSH_TARGET' --robot-namespace '$BRIDGE_NAMESPACE' --message-package '$BRIDGE_MESSAGE_PACKAGE' --setup '$HARU2_SSH_SETUP' --dialogue-json '$DIALOGUE_JSON' --dialogue-log '$DIALOGUE_LOG' --track-skeleton --skeleton-topic '$SKELETON_TOPIC' --track-interval-s 0.35 --skeleton-motion-json '$SKELETON_MOTION_JSON'" &
    BRIDGE_PID=$!
    ;;
  none)
    echo "[launcher] Haru2 bridge disabled; expecting another process to handle robot TTS/actions."
    ;;
  *)
    echo "[launcher] unknown HARU2_BRIDGE_MODE='$HARU2_BRIDGE_MODE' (expected docker, local, ssh, or none)" >&2
    exit 2
    ;;
esac

sleep 0.8
if [[ -n "${BRIDGE_PID}" ]] && ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
  echo "[launcher] Haru2 bridge exited during startup; stopping demo." >&2
  exit 1
fi
echo "[launcher] tailing dialogue log. Press Ctrl-C to stop all demo processes."
tail -f "$DIALOGUE_LOG" &
TAIL_PID=$!

wait "$MAIN_PID"
