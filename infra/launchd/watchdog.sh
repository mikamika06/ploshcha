#!/bin/bash
# Сторож публічного доступу до ПЛОЩІ.
#
# Ноутбук-сервер засинає (Low Power Sleep на батареї), і тунель падає разом із мережею. Прокинувшись,
# cloudflared перепідключається САМ, але з експоненційним відступом: заміряно 26 хвилин від
# пробудження до реєстрації. Саме в таке вікно ploshcha.org віддає 530/1033.
#
# Сторож скорочує це вікно до однієї хвилини: якщо ядро живе локально, а зовні сайт не відповідає
# двічі підряд — перезапускаємо тунель, а не чекаємо відступу.
LOG="$HOME/.ploshcha-watchdog.log"
LOCAL="http://127.0.0.1:8765/health"
PUBLIC="https://ploshcha.org/health"

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

curl -sf --max-time 5 "$LOCAL" >/dev/null || { say "ядро локально не відповідає — тунель ні до чого"; exit 0; }
ping -c 1 -W 2000 1.1.1.1 >/dev/null 2>&1 || { say "мережі немає — чекаємо"; exit 0; }

for i in 1 2; do
  curl -sf --max-time 10 "$PUBLIC" >/dev/null && exit 0
  sleep 5
done

say "публічно недоступно, ядро живе → перезапуск тунелю"
launchctl kickstart -k "gui/$(id -u)/org.ploshcha.tunnel" 2>>"$LOG" || say "kickstart не вдався"
