#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

set -e

LOG_DIR="/var/log/mindie"
LOG_FILE="${LOG_DIR}/install.log"
INSTALL_PATH_BASE="/usr/local/Ascend"
INSTALL_PATH="${INSTALL_PATH_BASE}/mindie"
SCRIPT_PATH="${INSTALL_PATH}/scripts"

function init_log()
{
    mkdir -p "$LOG_DIR" || {
        echo "WARN: Cannot write to $LOG_DIR, using local log."
        LOG_DIR="."
        LOG_FILE="./install.log"
    }
}

function print() {
    echo -e "[$(date '+%F %T')] [$1] $2" | tee -a "$LOG_FILE"
}

function parse_args()
{
    # parse args
    HAS_INSTALL_ARG=0
    HAS_UPGRADE_ARG=0
    QUIET_ARG=0
    EXTRA_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --quiet)
                QUIET_ARG=1
                ;;
            --upgrade)
                HAS_UPGRADE_ARG=1
                EXTRA_ARGS+=("$1")
                ;;
            --install)
                HAS_INSTALL_ARG=1
                ;;
            --install-path=*)     # form: --install-path=/opt/x
                INSTALL_PATH_BASE="${1#*=}"
                INSTALL_PATH="${INSTALL_PATH_BASE}/mindie"
                SCRIPT_PATH="${INSTALL_PATH}/scripts"
                ;;
            --install-path)        # form: --install-path /opt/x
                if [[ -z "${2:-}" ]]; then
                    echo "ERROR: --install-path requires a value" >&2
                    exit 1
                fi
                INSTALL_PATH_BASE="$2"
                INSTALL_PATH="${INSTALL_PATH_BASE}/mindie"
                SCRIPT_PATH="${INSTALL_PATH}/scripts"

                EXTRA_ARGS+=("$1=$2")
                shift
                ;;
            *)
                ;;
        esac
        shift
    done
}


function confirm_eula()
{
    eula_file=./eula_en.txt
    print "INFO" "show ${eula_file}"
    cat "${eula_file}" 1>&2
    read -n1 -re -p "Do you accept the EULA to MindIE?[Y/N]" answer
    case $answer in
        Y|y)
            print "INFO" "Accept EULA, continue..."
            ;;
        *)
            print "ERROR" "Reject EULA, exit!"
            exit 1
            ;;
    esac
}


function install_motor()
{
    motor_run=$(ls Ascend-mindie-service*.run 2>/dev/null | head -n 1)
    # install motor
    if [[ -z "$motor_run" ]]; then
        print "WARN" "Missing mindie-motor .run package. Skip."
    else
        print "INFO" "Installing Motor: $motor_run"
        chmod +x "$motor_run"

        MOTOR_ARGS=("${EXTRA_ARGS[@]}")

        # Add install args ONLY IF user provided --install
        if [[ $HAS_INSTALL_ARG -eq 1 ]]; then
            mkdir -p "$INSTALL_PATH"
            MOTOR_ARGS+=(--install --install-path="$INSTALL_PATH")
        fi

        echo "Running motor installer with args:"
        printf '  %q' "${MOTOR_ARGS[@]}"; echo

        bash "./$motor_run" "${MOTOR_ARGS[@]}"

        status=${PIPESTATUS[0]}
        ((status == 0)) || { print "ERROR" "Motor install failed"; exit $status; }
    fi
}


function install_llm()
{
    llm_run=$(ls Ascend-mindie-llm*.run 2>/dev/null | head -n 1)
    # install llm
    if [[ -z "$llm_run" ]]; then
        print "WARN" "Missing mindie-llm .run package. Skip."
    else
        print "INFO" "Installing LLM: $llm_run"
        chmod +x "$llm_run"

        LLM_ARGS=("${EXTRA_ARGS[@]}")

        # Add install args ONLY IF user provided --install
        if [[ $HAS_INSTALL_ARG -eq 1 ]]; then
            mkdir -p "$INSTALL_PATH"
            LLM_ARGS+=(--install --install-path="${INSTALL_PATH}")
        fi

        echo "Running llm installer with args:"
        printf '  %q' "${LLM_ARGS[@]}"; echo

        bash "./$llm_run" "${LLM_ARGS[@]}"

        status=${PIPESTATUS[0]}
        ((status == 0)) || { print "ERROR" "LLM install failed"; exit $status; }

    fi
}


function install_sd()
{
    wheel_file=$(ls mindiesd-*.whl 2>/dev/null | head -n 1)
    # install wheel
    if [[ -z "$wheel_file" ]]; then
        print "WARN" "Missing mindiesd .whl package. Skip."
    else
        python_bin=$(command -v python3 || true)
        python_bin=${python_bin:-$(command -v python || true)}

        if [[ -z "$python_bin" ]]; then
            print "ERROR" "No python found!"
            exit 1
        fi

        print "INFO" "Installing Python wheel: $wheel_file"
        "$python_bin" -m pip install "$wheel_file" --no-deps --log "$LOG_FILE" --force-reinstall \
            2>&1 | tee -a "$LOG_FILE"
    fi
}


function install_others()
{
    print "INFO" "Copy uninstall script"
    if [[ ! -f uninstall.sh ]]; then
        print "ERROR" "uninstall.sh not found!"
        exit 1
    fi
    cp set_env.sh "$INSTALL_PATH"
    cp uninstall.sh "$SCRIPT_PATH/uninstall.sh"
    chmod +x "$SCRIPT_PATH/uninstall.sh"
}


function main()
{
    init_log
    parse_args "$@"
    echo "-- $HAS_INSTALL_ARG  $HAS_UPGRADE_ARG"
    if [[ ( "$HAS_INSTALL_ARG" -eq 1 || "$HAS_UPGRADE_ARG" -eq 1 ) ]]; then
        if [[ "$QUIET_ARG" -ne 1 ]]; then
            confirm_eula
        fi

        mkdir -p "$INSTALL_PATH" "$SCRIPT_PATH"
        print "INFO" "Installing MindIE to: $INSTALL_PATH"

        install_llm
        install_motor
        install_sd
        install_others

        print "INFO" "MindIE installation SUCCESS!"
        print "INFO" "To uninstall: sudo bash $SCRIPT_PATH/uninstall.sh"
    fi
}

# Run main
main "$@"

