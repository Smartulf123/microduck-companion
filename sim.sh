#!/bin/bash
# Runs inside WSL Ubuntu. Starts Pollen's simulated duck (real robotd + MuJoCo body) and bridges
# robotd's unix socket to TCP 127.0.0.1:7810, where duck_adapter.py connects from Windows.
# Ctrl-C stops everything.
set -e
. ~/.cargo/env
SIM=~/Pollen/microduck/scripts/duck-sim
trap '$SIM down' EXIT
$SIM up
echo "bridge: 127.0.0.1:7810 -> robotd (Ctrl-C to stop the duck)"
socat TCP-LISTEN:7810,bind=127.0.0.1,reuseaddr,fork UNIX-CONNECT:$HOME/.cache/duck-sim/duck.sock
