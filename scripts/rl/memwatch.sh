#!/usr/bin/env bash
# Log anonymous/file RSS of RL trainer processes and the unit's cgroup memory every 15 s.
unit=${1:?unit}; out=${2:?log}
while systemctl --user is-active -q "$unit"; do
  line="$(date +%T)"
  for pid in $(pgrep -f "qwen_vl.rl.train_grpo" | head -4); do
    a=$(awk '/RssAnon/{print $2}' /proc/$pid/status 2>/dev/null); f=$(awk '/RssFile/{print $2}' /proc/$pid/status 2>/dev/null)
    line+=" pid$pid anon=$((a/1024/1024))G file=$((f/1024/1024))G"
  done
  cg=$(systemctl --user show "$unit" -p MemoryCurrent --value)
  line+=" unit=$((cg/1024/1024/1024))G host_avail=$(free -g | awk '/Mem/{print $7}')G"
  echo "$line" >> "$out"; sleep 15
done
