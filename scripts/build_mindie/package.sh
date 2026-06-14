#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

set -e

INPUT_DIR=$1
OUTPUT_DIR=$2
MAKESELF_FILE=$3

if [ -z "${INPUT_DIR:-}" ] || [ -z "${OUTPUT_DIR:-}" ] || [ -z "${MAKESELF_FILE:-}" ]; then
    echo "Error: Fail to create mindie package."
    echo "Usage: bash package.sh [INPUT_DIR] [OUTPUT_DIR] [MAKESELF_FILE]"
    exit 1
fi

if [ ! -f "$MAKESELF_FILE" ]; then
    echo "Error: makeself binary not found: $MAKESELF_FILE"
    exit 1
fi

if [ -z "$PY_TAG" ]; then
    echo "Warning: environment variable 'PY_TAG' is not set, try use the current version."
    PY_TAG=$(python3 -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')
fi
echo "Set PY_TAG as '${PY_TAG}'"

if [ -z "$PLATFORM" ]; then
    PLATFORM=$(arch)
fi
echo "Set PLATFORM as '${PLATFORM}'"

CUR_DIR=$(realpath "$(dirname "$0")")
MAKESELF_HEADER_FILE="$CUR_DIR/../makeself-header.sh"

# Generate timestamp directory name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PACKAGE_TMP_DIR="$CUR_DIR/.package_${TIMESTAMP}"

# Ensure directory is fresh
mkdir -p "$PACKAGE_TMP_DIR"

# Cleanup on exit (success or failure)
cleanup() {
    echo "Cleaning temporary directory: $PACKAGE_TMP_DIR"
    rm -rf "$PACKAGE_TMP_DIR"
}
trap cleanup EXIT

# Create metadata files
cat <<EOF > "$PACKAGE_TMP_DIR/version.info"
mindie: ${MINDIE_VERSION:-unknown}
mindie-motor: ${MOTOR_VERSION:-unknown}
mindie-llm: ${LLM_VERSION:-unknown}
mindie-sd: ${SD_VERSION:-unknown}
platform: ${PLATFORM:-$ARCH}
EOF

echo "Version Info:"
cat "$PACKAGE_TMP_DIR/version.info"

mkdir -p "$PACKAGE_TMP_DIR/scripts"
cp "$CUR_DIR/uninstall.sh" "$PACKAGE_TMP_DIR/scripts/"
cp "$CUR_DIR/set_env.sh" "$PACKAGE_TMP_DIR/"
cp "$CUR_DIR/install.sh" "$PACKAGE_TMP_DIR/"
cp "$CUR_DIR/uninstall.sh" "$PACKAGE_TMP_DIR/"
cp "$CUR_DIR/eula_ch.txt" "$PACKAGE_TMP_DIR/"
cp "$CUR_DIR/eula_en.txt" "$PACKAGE_TMP_DIR/"
# Copy input files to include in the package
cp -rf "$INPUT_DIR"/* "$PACKAGE_TMP_DIR/"


# Output file
mkdir -p "$OUTPUT_DIR"
OUT_FILE="${OUTPUT_DIR}/Ascend-mindie_${MINDIE_VERSION:-unknown}_${PY_TAG}_linux-${PLATFORM:-$(arch)}_abi${USE_CXX11_ABI:-0}.run"

echo "Building installer → $OUT_FILE"

# Run makeself
bash "$MAKESELF_FILE" \
    --header "$MAKESELF_HEADER_FILE" \
    --help-header "$CUR_DIR/help.info" \
    --gzip --complevel 4 --nomd5 --sha256 \
    --chown \
    "$PACKAGE_TMP_DIR/" \
    "${OUT_FILE}" \
    "Ascend-mindie-llm" \
    ./install.sh

echo "Generate $OUT_FILE successfully"
