#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

path="${BASH_SOURCE[0]}"

if [[ -f "$path" ]] && [[ "$path" =~ set_env\.sh ]]; then
    mindie_path=$(cd $(dirname $path); pwd )
    if [[ -f "${mindie_path}/mindie-llm/latest/version.info" ]]; then
    	if [[ -f "${mindie_path}/mindie-llm/latest/set_env.sh" ]]; then
			source ${mindie_path}/mindie-llm/latest/set_env.sh
		else
			echo "mindie-llm package is incomplete please check it."
		fi

    	if [[ -f "${mindie_path}/mindie-motor/set_env.sh" ]]; then
			source ${mindie_path}/latest/mindie-motor/set_env.sh
        elif [[ -f "${mindie_path}/mindie-service/set_env.sh" ]]; then
            # The old name of mindie-motor is mindie-service
			source ${mindie_path}/latest/mindie-service/set_env.sh
		else
			echo "mindie-motor package is incomplete please check it."
		fi
    else
        echo "The package of mindie is incomplete, please check it."
    fi
else
    echo "There is no 'set_env.sh' to import"
fi
