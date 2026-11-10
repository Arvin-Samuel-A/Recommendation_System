#!/usr/bin/env bash
# Serve the on-device demo (demo/mobile/) to phones on the same Wi-Fi.
#
# The phone downloads the page, the ONNX model and the movie data once; every
# recommendation is then computed on the phone itself (onnxruntime-web, WebAssembly).
# The phone also needs internet access once, to load onnxruntime-web from jsDelivr.
#
# Like run_demo.sh, it always runs in the main checkout, where the processed data and the
# run12 checkpoints live. The assets are exported on first use (scripts/export_mobile.py).
#
# Usage:
#   scripts/run_mobile_demo.sh             # port 8080
#   scripts/run_mobile_demo.sh 9000        # another port
#   scripts/run_mobile_demo.sh --export    # re-export the assets first
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here" && cd "$(git rev-parse --git-common-dir)/.." && pwd)"
cd "$root"

port=8080
export_assets=0
for arg in "$@"; do
  case "$arg" in
    --export) export_assets=1 ;;
    *) port="$arg" ;;
  esac
done

assets=demo/mobile/assets
if [ "$export_assets" = 1 ] || [ ! -e "$assets/model.onnx" ] || [ ! -e "$assets/items.json" ] || [ ! -e "$assets/users.json" ]; then
  uv run python scripts/export_mobile.py
fi

ip="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || true)"
echo
echo "On your phone (same Wi-Fi), open:  http://${ip:-<this-computer-IP>}:${port}/"
echo "On this computer:                  http://127.0.0.1:${port}/"
echo "(If the phone can't connect: allow Python in the macOS firewall prompt, or the Wi-Fi may"
echo " block device-to-device traffic -- connect this Mac to the phone's hotspot instead.)"
echo
exec uv run python -m http.server "$port" --bind 0.0.0.0 --directory demo/mobile
