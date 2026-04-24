#!/usr/bin/env bash
# Batch-downscale videos to 480p and crop off the top 1/3.
#
# Usage:
#   ./downscale_videos.sh <input_dir> <output_dir> [scaler] [crf]
#
# Examples:
#   ./downscale_videos.sh ./raw ./out                    # lanczos, crf 18
#   ./downscale_videos.sh ./raw ./out area 20            # "area" = best for big downscales
#
# Pipeline per video:
#   1. crop=in_w : in_h*2/3 : 0 : in_h/3   (drop the upper third)
#   2. scale=-2:480                        (height 480, width auto, even)
#
# Notes:
#   - scaler choices: lanczos (sharp, default), bicubic, bilinear, area, neighbor.
#   - crf: 0 = lossless, 18 ≈ visually lossless, 23 = default, 28 = small/ugly.
#   - fps is preserved (source is ~60 fps).
#   - audio is stripped (-an).

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <input_dir> <output_dir> [scaler=lanczos] [crf=18]"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
SCALER="${3:-lanczos}"
CRF="${4:-18}"

VF="crop=in_w:in_h*2/3:0:in_h/3,scale=-2:480:flags=${SCALER}"

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob nocaseglob
VIDEOS=("$INPUT_DIR"/*.{mp4,mov,avi,mkv,m4v})
shopt -u nocaseglob

if [[ ${#VIDEOS[@]} -eq 0 ]]; then
    echo "No videos found in $INPUT_DIR"
    exit 1
fi

echo "Found ${#VIDEOS[@]} videos. target=480p, crop=top-1/3 removed, scaler=${SCALER}, crf=${CRF}"
echo

i=0
for src in "${VIDEOS[@]}"; do
    i=$((i+1))
    name="$(basename "$src")"
    stem="${name%.*}"
    dst="$OUTPUT_DIR/${stem}_480p_cropped.mp4"

    echo "[$i/${#VIDEOS[@]}] $name  ->  $(basename "$dst")"

    if [[ -f "$dst" ]]; then
        echo "  (skip: already exists)"
        continue
    fi

    ffmpeg -hide_banner -loglevel warning -stats -y \
        -i "$src" \
        -vf "$VF" \
        -c:v libx264 -preset medium -crf "$CRF" \
        -pix_fmt yuv420p \
        -an \
        "$dst"
done

echo
echo "Done. Output in: $OUTPUT_DIR"
