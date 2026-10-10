#!/bin/bash
# تشغيل خدمة الإملاء الذكي (يتأكد أولاً أنها ليست شغالة)
PIDFILE=/tmp/smart_whisper.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    notify-send "🎤 الإملاء الذكي" "يعمل بالفعل ✓" 2>/dev/null
    exit 0
fi
# تأكد أن ydotoold يعمل (للكتابة التلقائية)
export YDOTOOL_SOCKET="$HOME/.ydotool_socket"
if ! pgrep -f "ydotoold --socket-path" >/dev/null; then
    nohup ydotoold --socket-path "$YDOTOOL_SOCKET" >/tmp/ydotoold.log 2>&1 &
    sleep 2
fi
cd "$HOME/Documents/whisper" || exit 1
export LD_LIBRARY_PATH=$(echo .venv/lib/python3.14/site-packages/nvidia/*/lib | tr ' ' ':'):$LD_LIBRARY_PATH
nohup .venv/bin/python -u smart_whisper_google_optimized.py >/tmp/whisper-daemon.log 2>&1 &
echo "بدأ التشغيل... انتظر تحميل النموذج (~20 ثانية)"
for i in $(seq 1 30); do
    grep -q "جاهز!" /tmp/whisper-daemon.log 2>/dev/null && break
    sleep 2
done
if grep -q "جاهز!" /tmp/whisper-daemon.log 2>/dev/null; then
    notify-send "🎤 الإملاء الذكي" "جاهز! اضغط F8 للبدء 🟢" 2>/dev/null
    echo "جاهز!"
else
    notify-send "🎤 الإملاء الذكي" "فشل التشغيل - راجع السجل" 2>/dev/null
    echo "فشل - راجع /tmp/whisper-daemon.log"
    tail -n 5 /tmp/whisper-daemon.log
fi
