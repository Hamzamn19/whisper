#!/bin/bash
# Toggle Smart Whisper recording via PID file (safe: never matches itself)
PIDFILE=/tmp/smart_whisper.pid
if [ -f "$PIDFILE" ]; then
    kill -USR1 "$(cat "$PIDFILE")" 2>/dev/null && exit 0
fi
# fallback if pidfile missing
pkill -USR1 -x smart_whisper 2>/dev/null
echo "whisper daemon not running?" >&2
exit 1
