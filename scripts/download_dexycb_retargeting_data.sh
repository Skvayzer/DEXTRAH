#!/usr/bin/env bash
# Download the official DexYCB subject archives needed by the offline retargeter.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 DATASET_ROOT [GDOWN_EXECUTABLE]" >&2
    exit 2
fi

dataset_root="$1"
gdown_executable="${2:-gdown}"
download_dir="${dataset_root}/downloads"
data_dir="${dataset_root}/data"
mkdir -p "$download_dir" "$data_dir"

# File IDs are the links published at https://dex-ycb.github.io/.
subjects=(
    "1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk 20200709-subject-01.tar.gz"
    "1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62 20200813-subject-02.tar.gz"
    "1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq 20200820-subject-03.tar.gz"
    "14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA 20200903-subject-04.tar.gz"
    "1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS 20200908-subject-05.tar.gz"
    "1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ 20200918-subject-06.tar.gz"
    "1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0 20200928-subject-07.tar.gz"
    "1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv 20201002-subject-08.tar.gz"
    "1j0BLkaCjIuwjakmywKdOO9vynHTWR0UH 20201015-subject-09.tar.gz"
    "1FvFlRfX-p5a5sAWoKEGc17zKJWwKaSB- 20201022-subject-10.tar.gz"
)

for entry in "${subjects[@]}"; do
    read -r file_id filename <<< "$entry"
    archive="${download_dir}/${filename}"
    subject_dir="${data_dir}/${filename%.tar.gz}"
    if [[ ! -f "$archive" ]]; then
        "$gdown_executable" "$file_id" -O "$archive"
    fi
    if [[ ! -d "$subject_dir" ]]; then
        tar -xzf "$archive" -C "$data_dir"
    fi
done

echo "DexYCB subjects are available under ${data_dir}"
