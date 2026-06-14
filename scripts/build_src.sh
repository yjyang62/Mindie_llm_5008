#!/bin/bash
function fn_build_src()
{
    cd $BUILD_DIR

    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    cmake .. $COMPILE_OPTIONS
    if [ "$USE_VERBOSE" == "ON" ];then
        cmake --build . --target all -- VERBOSE=1 -j"$thread_num"
    else
        cmake --build . --target all -- -j"$thread_num"
    fi
    cmake --install .
    cd -
}