#!/bin/bash
# إيقاف خدمة الإملاء الذكي
PIDFILE=/tmp/smart_whisper.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")" 2>/dev/null
    rm -f "$PIDFILE"
    notify-send "🎤 الإملاء الذكي" "تم الإيقاف 🔴" 2>/dev/null
    echo "تم الإيقاف"
else
    pkill -f smart_whisper_google_optimized 2>/dev/null
    notify-send "🎤 الإملاء الذكي" "لم يكن يعمل" 2>/dev/null
    echo "لم يكن يعمل"
fi
