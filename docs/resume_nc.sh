#!/bin/bash
# Resume-download latest_net_NC.pth until it reaches the exact expected size.
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890

ID="1SgiK6fBHXd2yICBhNNfoncUWjSUxslYe"
OUT="latest_net_NC.pth"
TARGET=37238739
cd "$(dirname "$0")"

fresh_token() {
  local html uuid
  html=$(curl -sL -m 90 -c /tmp/gc_nc.txt "https://drive.google.com/uc?export=download&id=$ID")
  uuid=$(printf '%s' "$html" | grep -o 'name="uuid" value="[^"]*"' | sed 's/.*value="//;s/"//')
  if [ -z "$uuid" ]; then uuid="92f93260-ebb5-480f-9e67-56ec73eee703"; fi
  URL="https://drive.usercontent.google.com/download?id=$ID&export=download&confirm=t&uuid=$uuid"
}

for attempt in $(seq 1 25); do
  cur=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  if [ "$cur" -ge "$TARGET" ]; then
    echo "DONE size=$cur target=$TARGET"
    break
  fi
  echo "[attempt $attempt] have=$cur need=$((TARGET-cur))"
  fresh_token
  curl -sL -C - -m 420 -b /tmp/gc_nc.txt -o "$OUT" "$URL"
  new=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  echo "[attempt $attempt] now=$new (+$((new-cur)))"
  if [ "$new" -le "$cur" ]; then
    echo "  no progress, sleeping 10s"
    sleep 10
  fi
done

final=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
echo "=== FINAL size=$final target=$TARGET ==="
if [ "$final" -eq "$TARGET" ]; then echo "SIZE MATCH: OK"; else echo "SIZE MISMATCH: INCOMPLETE"; fi
