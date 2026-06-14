#!/bin/bash
function fn_build_version_info()
{
    version=${MINDIE_LLM_VERSION_OVERRIDE:-1.0.0}
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    commit_id=$(git rev-parse HEAD)
    touch $OUTPUT_DIR/version.info
    cat>$OUTPUT_DIR/version.info<<EOF
    MindIE-LLM Version : ${version}
    Platform : ${ARCH}
    branch : ${branch}
    commit id : ${commit_id}
EOF
}

function get_version()
{
    # 黄区
    VERSION="1.0.0"
    if [ ! -f "${CODE_ROOT}"/../CI/config/version.ini ]; then
        echo "version.ini is not existed, failed to get version info"
    else
        VERSION=$(cat ${CODE_ROOT}/../CI/config/version.ini | grep "PackageName" | cut -d "=" -f 2)
        echo $VERSION
    fi
    export MINDIE_VERSION=$VERSION
}