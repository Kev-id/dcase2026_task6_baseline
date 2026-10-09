#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="${PROJECT_ROOT}/downloads/clotho-moment"
FEATURE_DIR="${PROJECT_ROOT}/features/clotho-moment"
BASE_URL="https://huggingface.co/datasets/lighthouse-emnlp2024/Clotho-Moment_CLAP_features/resolve/main"

mkdir -p "${DOWNLOAD_DIR}" "${FEATURE_DIR}"

download_file() {
    local filename="$1"
    local total_size="$2"
    local part_count="$3"
    local target="${DOWNLOAD_DIR}/${filename}"
    local part_dir="${DOWNLOAD_DIR}/.${filename}.parts"
    local url="${BASE_URL}/${filename}"

    if [[ -f "${target}" ]] && [[ "$(stat -c %s "${target}")" -eq "${total_size}" ]]; then
        echo "${filename}: complete file already exists"
        return 0
    fi

    mkdir -p "${part_dir}"
    echo "${filename}: downloading ${total_size} bytes in ${part_count} parts"

    local -a pids=()
    local index
    for ((index = 0; index < part_count; index++)); do
        (
            local start=$((total_size * index / part_count))
            local end=$((total_size * (index + 1) / part_count - 1))
            local expected=$((end - start + 1))
            local part
            part="${part_dir}/$(printf 'part_%03d' "${index}")"
            touch "${part}"

            local attempt=0
            while true; do
                local have
                have="$(stat -c %s "${part}")"

                if [[ "${have}" -eq "${expected}" ]]; then
                    echo "${filename} part ${index}: complete"
                    break
                fi
                if [[ "${have}" -gt "${expected}" ]]; then
                    echo "${filename} part ${index}: oversized (${have} > ${expected})" >&2
                    exit 1
                fi

                attempt=$((attempt + 1))
                if [[ "${attempt}" -gt 20 ]]; then
                    echo "${filename} part ${index}: too many retries" >&2
                    exit 1
                fi

                local request_start=$((start + have))
                echo "${filename} part ${index}: bytes ${request_start}-${end}, attempt ${attempt}"
                curl --fail --silent --show-error --location \
                    --connect-timeout 30 --max-time 7200 \
                    --range "${request_start}-${end}" \
                    "${url}" >> "${part}" || sleep 3
            done
        ) &
        pids+=("$!")
    done

    local failed=0
    local pid
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "${filename}: at least one part failed" >&2
        return 1
    fi

    local assembled="${target}.assembling"
    : > "${assembled}"
    for ((index = 0; index < part_count; index++)); do
        cat "${part_dir}/$(printf 'part_%03d' "${index}")" >> "${assembled}"
    done

    local assembled_size
    assembled_size="$(stat -c %s "${assembled}")"
    if [[ "${assembled_size}" -ne "${total_size}" ]]; then
        echo "${filename}: assembled size ${assembled_size}, expected ${total_size}" >&2
        return 1
    fi

    mv "${assembled}" "${target}"
    rm -rf "${part_dir}"
    echo "${filename}: assembled successfully"
}

download_file "clap.tar.gz" 20775803739 12 &
audio_pid=$!
download_file "clap_text.tar.gz" 1967280822 2 &
text_pid=$!

failed=0
wait "${audio_pid}" || failed=1
wait "${text_pid}" || failed=1
if [[ "${failed}" -ne 0 ]]; then
    echo "Download failed; rerun this script to resume existing parts." >&2
    exit 1
fi

echo "Extracting audio features..."
tar -xzf "${DOWNLOAD_DIR}/clap.tar.gz" -C "${FEATURE_DIR}"

echo "Extracting text features..."
tar -xzf "${DOWNLOAD_DIR}/clap_text.tar.gz" -C "${FEATURE_DIR}"

audio_count="$(find "${FEATURE_DIR}/clap" -maxdepth 1 -type f -name '*.npz' | wc -l)"
text_count="$(find "${FEATURE_DIR}/clap_text" -maxdepth 1 -type f -name '*.npz' | wc -l)"

echo "Audio feature files: ${audio_count}"
echo "Text feature files: ${text_count}"

if [[ "${audio_count}" -ne 51240 ]] || [[ "${text_count}" -ne 44261 ]]; then
    echo "Unexpected feature count" >&2
    exit 1
fi

echo "Clotho-Moment preparation complete."
