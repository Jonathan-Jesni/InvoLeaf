#!/usr/bin/env bash
# Download only the images the committed splits use, via git, through a flaky proxy.
#
#   bash scripts/fetch_data.sh
#
# Why not a plain clone: the cloud instances reach GitHub through an egress proxy that
# refuses connections in bursts (HTTP 503) and stalls long transfers. A normal clone of
# PlantVillage is one 2 GB stream with no resume, so a stall throws everything away. The
# opposite - letting git fetch each image on demand - means ~29,000 separate requests,
# each one a chance to be refused.
#
# This does what git's partial clone does internally, but in batches:
#   1. clone only the file list (blob:none, a few MB),
#   2. list the image blobs under the folders we use,
#   3. fetch them BATCH at a time, each batch retried on refusal or stall,
#   4. check the folders out locally, with no further network access.
#
# Re-running is safe and cheap: blobs already downloaded are skipped.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BATCH="${BATCH:-400}"   # blobs per request; ~8 MB for PlantVillage-sized JPEGs
TRIES="${TRIES:-10}"    # attempts per clone or batch before moving on

# Abort a request that drops below 10 KB/s for 20 s, so a stalled batch is retried
# instead of hanging forever. Overrides any global lowSpeed settings.
GIT_NET=(-c http.lowSpeedLimit=10000 -c http.lowSpeedTime=20 -c core.quotePath=false)

fetch_repo () {         # fetch_repo <url> <dest> <path-regex> <checkout-pathspec>...
  local url="$1" dest="$2" regex="$3"; shift 3
  echo "=== ${dest} ==="

  if [[ ! -d "${dest}/.git" ]]; then
    local i
    for i in $(seq 1 "$TRIES"); do
      git "${GIT_NET[@]}" clone --depth 1 --filter=blob:none --no-checkout "$url" "$dest" \
        && break
      rm -rf "$dest"; echo "   clone refused, retry ${i}/${TRIES}"; sleep 15
    done
    [[ -d "${dest}/.git" ]] || { echo "   could not clone ${url}"; return 1; }
  fi

  local work; work=$(mktemp -d)

  # Wanted blobs: the tree is complete in a blob:none clone, so this needs no network.
  git -C "$dest" -c core.quotePath=false ls-tree -r HEAD \
    | awk -F'\t' -v re="$regex" '$2 ~ re { split($1, a, " "); print a[3] }' \
    | sort -u > "${work}/wanted"

  # Blobs already on disk. --batch-all-objects only enumerates local objects, so unlike
  # a per-object lookup it can never trigger git's one-at-a-time lazy fetch.
  git -C "$dest" cat-file --batch-all-objects --batch-check='%(objectname)' \
    | sort -u > "${work}/present"
  comm -23 "${work}/wanted" "${work}/present" > "${work}/missing"

  local wanted missing
  wanted=$(wc -l < "${work}/wanted"); missing=$(wc -l < "${work}/missing")
  echo "   ${wanted} images wanted, ${missing} still to download"

  if (( missing > 0 )); then
    split -l "$BATCH" "${work}/missing" "${work}/part."
    local parts n=0 part i
    parts=$(ls "${work}"/part.* | wc -l)
    for part in "${work}"/part.*; do
      n=$((n + 1))
      for i in $(seq 1 "$TRIES"); do
        # The same command git's promisor fetch runs, but for a whole batch of objects.
        if git "${GIT_NET[@]}" -C "$dest" -c fetch.negotiationAlgorithm=noop \
             fetch -q origin --no-tags --no-write-fetch-head --recurse-submodules=no \
             --filter=blob:none --stdin < "$part"; then
          echo "   batch ${n}/${parts} ok"; break
        fi
        echo "   batch ${n}/${parts} refused or stalled, retry ${i}/${TRIES}"; sleep 10
      done
    done
  fi

  # Local checkout of just our folders. Any straggler that never arrived would be
  # fetched individually here, which is fine for a handful.
  git "${GIT_NET[@]}" -C "$dest" checkout HEAD -- "$@" \
    || echo "   checkout incomplete - rerun this script"
  rm -rf "$work"
}

fetch_repo https://github.com/spMohanty/PlantVillage-Dataset data/PlantVillage-Dataset \
  '^raw/color/(Tomato|Corn|Apple|Grape)' \
  'raw/color/Tomato*' 'raw/color/Corn*' 'raw/color/Apple*' 'raw/color/Grape*'

fetch_repo https://github.com/pratikkayal/PlantDoc-Dataset data/PlantDoc-Dataset \
  '^(train|test)/Tomato' \
  'train/Tomato*' 'test/Tomato*'

echo
echo "=== counts (expect 29245 and about 726) ==="
find data/PlantVillage-Dataset/raw/color -type f 2>/dev/null | wc -l
find data/PlantDoc-Dataset -path '*/.git' -prune -o -type f -print 2>/dev/null | wc -l
echo "If a count is short, run this script again - it only fetches what is missing."
