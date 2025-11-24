#!/usr/bin/env bash
set -euo pipefail

echo
echo "======================================="
echo "   MachineInnovators – Clean Up All"
echo "======================================="
echo

# 1) ferma e rimuove stack + volumi
echo "→ docker compose down -v…"
docker compose down -v || true

# 2) porta 5000/8080/8000/9090/9091
echo "→ libero porte comuni (5000, 8080, 8000, 9090, 9091)…"
for p in 5000 8080 8000 9090 9091; do
  if command -v fuser >/dev/null 2>&1; then
    sudo fuser -k "${p}/tcp" 2>/dev/null || true
  fi
done

echo "✓ Clean completo."
