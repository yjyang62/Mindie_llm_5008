#!/bin/bash
set -e
VERSION=VERSION_PLACEHOLDER
LOG_PATH=LOG_PATH_PLACEHOLDER
LOG_NAME=LOG_NAME_PLACEHOLDER
MAX_LOG_SIZE=$((1024*1024*50))

ori_umsk=$(umask)
umask 0027 # The permission mask for creating log files(640) and directories(750).

function exit_solver() {
    if [ -f "$log_file" ]; then
        chmod 440 ${log_file}
    fi
    if [ -d "$LOG_PATH" ]; then
        chmod 750 ${LOG_PATH}
    fi
    exit_code=$?
    if [ ${exit_code} -ne 0 ];then
        print "ERROR" "Uninstall failed, [ERROR] ret code:${exit_code}"
        exit ${exit_code}
    fi
    exit 0
}

trap exit_solver EXIT

if [ "$UID" = "0" ]; then
    log_file=${LOG_PATH}${LOG_NAME}
else
    LOG_PATH="${HOME}${LOG_PATH}"
    log_file=${LOG_PATH}${LOG_NAME}
fi

function print() {
    if [ ! -f "$log_file" ]; then
        if [ ! -d "${LOG_PATH}" ];then
            mkdir -p ${LOG_PATH}
        fi
        touch $log_file
    fi
    chmod 750 ${LOG_PATH}
    chmod 640 ${log_file}
    if [ x"$log_file" = x ]; then
        echo -e "[mindie-llm] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
    else
        if [ $(stat -c %s $log_file) -gt $MAX_LOG_SIZE ];then 
            echo -e "[mindie-llm] [$(date +%Y%m%d-%H:%M:%S)] [$1] log file is bigger than $MAX_LOG_SIZE, stop write log to file"
            echo -e "[mindie-llm] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
        else
            echo -e "[mindie-llm] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2" | tee -a $log_file
        fi
    fi
    chmod 440 $log_file
}

CUR_DIR=$(dirname $(readlink -f $0))
cd $CUR_DIR/../../
chmod -R u+w ../mindie_llm

if [ -L "latest" ]; then
    rm -f latest
fi
if [ -d "${VERSION}" ]; then
    rm -rf $VERSION
fi
if [ -f "set_env.sh" ]; then
    rm -f set_env.sh
fi

chmod 640 ${log_file}
python3 -m pip uninstall mindie_llm --log-file ${log_file}

print "INFO" "Ascend-mindie-llm uninstall success!"

umask "$ori_umsk" # Restore the original default permission mask.
