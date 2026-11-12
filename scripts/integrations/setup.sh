#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

install_tools() {
  local directory="$1" url="$2" revision="$3" tools="$4"
  if [[ ! -d "$directory/.git" ]]; then
    git clone --depth 1 "$url" "$directory"
    if [[ "$(git -C "$directory" rev-parse HEAD)" != "$revision" ]]; then
      git -C "$directory" fetch --depth 1 origin "$revision"
      git -C "$directory" checkout --detach FETCH_HEAD
    fi
  fi
  local link="$directory/xiconwam_tools"
  if [[ ! -e "$link" && ! -L "$link" ]]; then
    ln -s "../../scripts/integrations/$tools" "$link"
  fi
}

install_tools third_party/OpenWAM https://github.com/OpenWAM-Official/OpenWAM.git 9bd3f87daa8c5376db42fd304ba17e65c5a888d7 openwam
install_tools third_party/VPP2 https://github.com/roboterax/video-prediction-policy-2.git a103a1e1784d3e45014549a82238894d3ed85959 vpp2
