#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/results/netem_delivery"
PYTHON_BIN="$(command -v python3 || command -v python)"
DURATION_S=180
REPETITIONS=3
PAYLOAD_BYTES=1024
PORT=18767

usage() {
  printf '%s\n' \
    "Usage: $0 [--output DIR] [--duration-s N] [--repetitions N] [--python PATH]" \
    "" \
    "Captures ideal/practical/degraded/stress UDP delivery through Linux tc netem," \
    "then evaluates the measured traces over the frozen trajectory bank."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT_ROOT="$(realpath -m "$2")"; shift 2 ;;
    --duration-s) DURATION_S="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$(realpath "$2")"; shift 2 ;;
    --payload-bytes) PAYLOAD_BYTES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  printf 'This harness requires Linux.\n' >&2
  exit 1
fi
for command_name in ip tc sudo; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$command_name" >&2
    exit 1
  fi
done
if ! sudo -v; then
  printf 'sudo or CAP_NET_ADMIN is required for network namespaces and tc netem.\n' >&2
  exit 1
fi

CURRENT_USER="$(id -un)"
RUN_TOKEN="${USER:-user}_$$_${RANDOM}"
NS_NAME="dtns_${RUN_TOKEN:0:10}"
HOST_IF="dth${RUN_TOKEN:0:8}"
NS_IF="dtn${RUN_TOKEN:0:8}"
HOST_IP="10.203.17.1"
NS_IP="10.203.17.2"
RECEIVER_PID=""
SUDO_KEEPALIVE_PID=""

cleanup() {
  if [[ -n "$RECEIVER_PID" ]] && kill -0 "$RECEIVER_PID" 2>/dev/null; then
    kill "$RECEIVER_PID" 2>/dev/null || true
    wait "$RECEIVER_PID" 2>/dev/null || true
  fi
  if [[ -n "$SUDO_KEEPALIVE_PID" ]] && kill -0 "$SUDO_KEEPALIVE_PID" 2>/dev/null; then
    kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
  fi
  sudo ip netns del "$NS_NAME" 2>/dev/null || true
  sudo ip link del "$HOST_IF" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(while true; do sudo -n -v 2>/dev/null || exit; sleep 60; done) &
SUDO_KEEPALIVE_PID=$!

cd "$REPO_ROOT"
"$PYTHON_BIN" -c "import numpy, pandas; import DigitalTwin.analysis.netem_transport"
"$PYTHON_BIN" -c "from DigitalTwin.analysis import service_timing_budget_study as s; paths=s.discover(); print(f'Frozen trajectory preflight: {len(paths)} trajectories')"

sudo ip netns add "$NS_NAME"
sudo ip link add "$HOST_IF" type veth peer name "$NS_IF"
sudo ip link set "$NS_IF" netns "$NS_NAME"
sudo ip addr add "${HOST_IP}/24" dev "$HOST_IF"
sudo ip link set "$HOST_IF" up
sudo ip netns exec "$NS_NAME" ip addr add "${NS_IP}/24" dev "$NS_IF"
sudo ip netns exec "$NS_NAME" ip link set lo up
sudo ip netns exec "$NS_NAME" ip link set "$NS_IF" up

# name|application rate Hz|netem delay ms|netem jitter ms|netem loss percent
CONDITIONS=(
  "ideal|10|0|0|0"
  "practical|5|50|20|1"
  "degraded|2|200|50|5"
  "stress|1|300|100|10"
)

mkdir -p "$OUTPUT_ROOT/captures"
for replicate in $(seq 1 "$REPETITIONS"); do
  for specification in "${CONDITIONS[@]}"; do
    IFS='|' read -r condition rate_hz delay_ms jitter_ms loss_percent <<< "$specification"
    capture_dir="$OUTPUT_ROOT/captures/$condition/replicate_$(printf '%02d' "$replicate")"
    mkdir -p "$capture_dir"

    if [[ "$condition" == "ideal" ]]; then
      sudo ip netns exec "$NS_NAME" tc qdisc replace dev "$NS_IF" root netem
    else
      sudo ip netns exec "$NS_NAME" tc qdisc replace dev "$NS_IF" root netem \
        delay "${delay_ms}ms" "${jitter_ms}ms" distribution normal \
        loss random "${loss_percent}%"
    fi
    sudo ip netns exec "$NS_NAME" tc -s qdisc show dev "$NS_IF" > "$capture_dir/tc_qdisc_before.txt"

    printf 'Capturing %s replicate %d: %s Hz, %s +/- %s ms, %s%% loss\n' \
      "$condition" "$replicate" "$rate_hz" "$delay_ms" "$jitter_ms" "$loss_percent"

    "$PYTHON_BIN" -m DigitalTwin.analysis.netem_transport receive \
      --bind "$HOST_IP" --port "$PORT" \
      --output "$capture_dir/received_packets.csv" --idle-timeout-s 5 &
    RECEIVER_PID=$!
    sleep 0.5

    sudo ip netns exec "$NS_NAME" sudo -u "$CURRENT_USER" \
      env "PATH=$PATH" "PYTHONPATH=$REPO_ROOT" \
      "$PYTHON_BIN" -m DigitalTwin.analysis.netem_transport send \
      --host "$HOST_IP" --port "$PORT" --rate-hz "$rate_hz" \
      --duration-s "$DURATION_S" --payload-bytes "$PAYLOAD_BYTES" \
      --output "$capture_dir/sent_packets.csv"

    wait "$RECEIVER_PID"
    RECEIVER_PID=""
    sudo ip netns exec "$NS_NAME" tc -s qdisc show dev "$NS_IF" > "$capture_dir/tc_qdisc_after.txt"
    "$PYTHON_BIN" -m DigitalTwin.analysis.netem_transport summarize \
      --sent "$capture_dir/sent_packets.csv" \
      --received "$capture_dir/received_packets.csv" \
      --output-dir "$capture_dir" --condition "$condition" \
      --replicate "$replicate" --rate-hz "$rate_hz" \
      --delay-ms "$delay_ms" --jitter-ms "$jitter_ms" \
      --loss-percent "$loss_percent" \
      --qdisc-record "$capture_dir/tc_qdisc_after.txt"
  done
done

"$PYTHON_BIN" -m DigitalTwin.analysis.netem_delivery_replay \
  --capture-root "$OUTPUT_ROOT/captures" \
  --output "$OUTPUT_ROOT/evidence"

printf '\nCompleted. Review:\n  %s\n' "$OUTPUT_ROOT/evidence/netem_delivery_summary.md"
