#!/bin/bash
function parse_args(){
    test_type=""
    test_target="all"
    enable_coverage="OFF"
    enable_clean="OFF"

    build_dir="$CODE_ROOT/build"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -T | --type)
                if [[ $# -gt 1 && $2 != "-"* ]]; then
                    test_type=$2
                    if [[ "$test_type" = "fuzz" ]]; then
                    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_FUZZ_TEST=ON"
                    fi
                    shift 2
                else
                    echo "Error: Argument required after -T | --type"
                    exit 1
                fi
                ;;
            -t | --target) 
                if [[ $# -gt 1 && $2 != "-"* ]]; then
                    test_target=$2
                    shift 2
                else
                    echo "Error: Argument required after -t | --target"
                    exit 1
                fi
                ;;
            -C | --coverage)
                enable_clean="ON"
                enable_coverage="ON"
                shift
                ;;
            --asan)  # Add ASan flag
                enable_asan="ON"
                shift
                ;;
            *)
                return;
                ;;
        esac
    done
}

function fn_dlt(){

    if [ "$enable_clean" == "ON" ]; then
        fn_clean
    fi

    if [ ! -d $build_dir ]; then
        mkdir -p $build_dir
    fi
    cd $build_dir
    if [[ "$enable_asan" == "ON" ]]; then
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_CXX_FLAGS=-fsanitize=address -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address"
    fi

    cmake .. $COMPILE_OPTIONS

    if [ "$test_target" == "all" ]; then
        cmake --build . --target all -j$thread_num
    else
        cmake --build . --target ${test_target}_${test_type} -j$thread_num
    fi
    if [[ $enable_coverage == "ON" ]]; then
        if find $CODE_ROOT -name "*.gcda" | grep -q .; then
            cmake --build $CODE_ROOT/build --target coverage -j$thread_num
        else
            echo "gcda file not exist. Skipping coverage report generation."
        fi
    fi
    cd -
}