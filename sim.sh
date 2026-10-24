#!/bin/bash
# Runs inside WSL Ubuntu: Pollen's real robot software (robotd + tofd) driving the MuJoCo body that
# start.ps1 runs on Windows, where the 3D view gets the GPU (WSL here only has software OpenGL).
# robotd's socket is bridged to TCP 127.0.0.1:7810 for duck_adapter.py. Ctrl-C stops it.
# The same daemons and arguments as Pollen's scripts/duck-sim, minus configd/updaterd.
set -e
REPO=~/Pollen/microduck
RL=~/Pollen/microduck_rl
STATE=~/.cache/duck-sim
BODY="$(ip route show default | awk '{print $3}'):7801"  # Windows, as WSL sees it
ORT="$(find "$RL/.venv/lib" -name 'libonnxruntime.so*' -type f | head -1)"
P="$STATE/policies/current"
mkdir -p "$STATE"

[ -f "$P/alpha_stand.onnx" ] || sh "$REPO/scripts/seed-policies.sh" "$STATE/policies"
[ -d "$STATE/sounds/duck-a" ] || DUCK_IDENTITY=duck-a "$REPO/target/debug/sounds" ensure-bank --dir "$STATE/sounds/duck-a" >/dev/null 2>&1 || true
cat > "$STATE/companion.toml" <<TOML
[policy]
enabled = true
walk = "$P/${DUCK_WALK:-velstand.onnx}"
stand = "${DUCK_STAND:-none}"
sitstand = "$P/alpha_sitstand.onnx"
ground_pick = "$P/alpha_ground_pick.onnx"
kick_left = "$P/ball_kick_left.onnx"
kick_right = "$P/ball_kick_right.onnx"
roulade = "$P/roulade.onnx"

[audio]
bank = "$STATE/sounds/duck-a"
device = "default"

[theremin]
socket = "$STATE/duck-a-tof.sock"
TOML

echo "waiting for the body on $BODY (start.ps1 starts it)"
until (exec 3<>"/dev/tcp/${BODY%:*}/${BODY#*:}") 2>/dev/null; do sleep 0.5; done

trap 'kill $(jobs -p) 2>/dev/null' EXIT
rm -f "$STATE/duck-a.sock" "$STATE/duck-a-tof.sock"
DUCK_RUNTIME_DIR="$STATE" "$REPO/target/debug/tofd" --sim "$BODY" --socket "$STATE/duck-a-tof.sock" \
    > "$STATE/duck-a-tof.log" 2>&1 &
DUCK_IDENTITY=duck-a DUCK_RUNTIME_DIR="$STATE" ORT_DYLIB_PATH="$ORT" RUST_LOG=info \
    "$REPO/target/debug/robotd" --sim "$BODY" --params "$STATE/companion.toml" \
    --socket "$STATE/duck-a.sock" > "$STATE/duck-a.log" 2>&1 &
until [ -S "$STATE/duck-a.sock" ]; do sleep 0.2; done

echo "robot software up — bridge 127.0.0.1:7810 -> robotd (Ctrl-C stops it)"
socat TCP-LISTEN:7810,bind=127.0.0.1,reuseaddr,fork UNIX-CONNECT:"$STATE/duck-a.sock"
