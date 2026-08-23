#!/usr/bin/env bash
# Перенесення ПЛОЩІ на домашній сервер: ядро + тунель під systemd.
#
# Ноутбук сервером бути не може — він засинає (виміряно: два вікна по 2 і 6 годин за одну добу,
# і ще 26 хвилин відступу cloudflared після пробудження). Сервер не спить, тож і сторож не потрібен:
# systemd тримає обидва процеси з Restart=always.
#
# Запуск із мака:  infra/server/deploy.sh [користувач@адреса]
set -euo pipefail

HOST="${1:-root@192.168.0.209}"
KEY="${PLOSHCHA_SSH_KEY:-$HOME/.ssh/dellserver_ed25519}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="/opt/ploshcha"
SSH=(ssh -i "$KEY" -o ConnectTimeout=10 "$HOST")

echo "== 1/6 перевірка сервера =="
"${SSH[@]}" 'uname -m; python3 --version; systemctl --version | head -1' || {
  echo "сервер недоступний: увімкни його й перевір, що він у тій самій мережі" >&2; exit 1; }

echo "== 2/6 код і збірка (без docs, .venv, node_modules) =="
rsync -az --delete -e "ssh -i $KEY" \
  --exclude '.venv' --exclude 'node_modules' --exclude 'docs' --exclude '.git' \
  --exclude 'eval/traces' --exclude '.ruff_cache' --exclude 'data' \
  "$ROOT/" "$HOST:$DEST/"

echo "== 3/6 стан села (база + сесії) — лише якщо на сервері ще порожньо =="
"${SSH[@]}" "test -f $DEST/data/ploshcha/ploshcha.db" \
  && echo "   база на сервері вже є — не чіпаю" \
  || rsync -az -e "ssh -i $KEY" "$ROOT/data/" "$HOST:$DEST/data/"

echo "== 4/6 секрети (окремо, права 600) =="
rsync -a -e "ssh -i $KEY" "$ROOT/.env" "$HOST:$DEST/.env"
rsync -a -e "ssh -i $KEY" "$HOME/.ploshcha-tunnel.token" "$HOST:$DEST/.tunnel-token"
"${SSH[@]}" "chmod 600 $DEST/.env $DEST/.tunnel-token"

echo "== 5/6 оточення, cloudflared, юніти =="
"${SSH[@]}" bash -s <<'REMOTE'
set -euo pipefail
DEST=/opt/ploshcha
cd "$DEST"

# Ubuntu ставить python3 без `ensurepip`, тож `python3 -m venv` падає на голій системі.
python3 -c 'import ensurepip' 2>/dev/null || { apt-get update -qq; apt-get install -y python3-venv; }
# Перевіряємо саме pip: зламане оточення (створене до встановлення `python3-venv`) має python,
# але не має pip, і тоді установка залежностей падає з «No such file or directory».
[ -x .venv/bin/pip ] || { rm -rf .venv; python3 -m venv .venv; }
./.venv/bin/pip -q install --upgrade pip
./.venv/bin/pip -q install "pydantic>=2.7" "openai>=1.40"

if ! command -v cloudflared >/dev/null; then
  ARCH=$(dpkg --print-architecture)
  curl -fsSL -o /tmp/cloudflared.deb \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH}.deb"
  dpkg -i /tmp/cloudflared.deb
fi

# Сервер — теж ноутбук: із закритою кришкою Ubuntu присипляє машину, і тунель ляже так само, як
# лягав на маку. Вимикаємо реакцію на кришку, доки він працює сервером.
sed -i 's/^#\?HandleLidSwitch=.*/HandleLidSwitch=ignore/; s/^#\?HandleLidSwitchExternalPower=.*/HandleLidSwitchExternalPower=ignore/' /etc/systemd/logind.conf
systemctl restart systemd-logind || true

cat > /etc/systemd/system/ploshcha-core.service <<'UNIT'
[Unit]
Description=ПЛОЩА — ядро (API + статика)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/ploshcha
ExecStart=/opt/ploshcha/.venv/bin/python /opt/ploshcha/services/sim/scripts/serve_ploshcha.py \
  --port 8765 --condition viche --resume --db /opt/ploshcha/data/ploshcha/ploshcha.db \
  --max-tokens 0 --max-usd 0 --max-items 0 --workers 6
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/ploshcha-tunnel.service <<'UNIT'
[Unit]
Description=ПЛОЩА — Cloudflare Tunnel
After=network-online.target ploshcha-core.service
Wants=network-online.target

[Service]
Type=simple
ExecStart=/bin/bash -c '/usr/bin/cloudflared tunnel --no-autoupdate run --token "$(cat /opt/ploshcha/.tunnel-token)"'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable ploshcha-core.service ploshcha-tunnel.service
# ★ Перезапускаємо ЛИШЕ ядро.
#
# Тунель не має стосунку до нашого коду, а його рестарт — це видима діра: поки новий конектор
# реєструється в Cloudflare, походження зникає й сайт віддає 530/1033. Саме так і сталось о 23:23 —
# «сайт упав» був наслідком мого ж викочування, а не мережі. Тунель піднімаємо лише якщо він лежить.
systemctl restart ploshcha-core.service
systemctl start ploshcha-tunnel.service
sleep 8
systemctl is-active ploshcha-core.service ploshcha-tunnel.service
curl -sf --max-time 5 http://127.0.0.1:8765/health >/dev/null && echo "   ядро на сервері відповідає"
REMOTE

echo "== 6/6 глушимо мак =="
for a in watchdog tunnel core; do
  launchctl unload "$HOME/Library/LaunchAgents/org.ploshcha.$a.plist" 2>/dev/null || true
done
pkill -f "cloudflared tunnel" 2>/dev/null || true
pkill -f serve_ploshcha.py 2>/dev/null || true

echo "== перевірка ззовні =="
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 https://ploshcha.org/health || true)
  [ "$code" = "200" ] && { echo "ploshcha.org 200 — тепер із сервера"; exit 0; }
  sleep 4
done
echo "ploshcha.org ще не піднявся; дивись: ssh $HOST journalctl -u ploshcha-tunnel -n 50" >&2
exit 1
