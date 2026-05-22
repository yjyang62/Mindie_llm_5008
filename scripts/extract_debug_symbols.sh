#!/bin/bash
function fn_extract_debug_symbols() {
    local input_dir=$1
    local output_dir=$2
    mkdir -p "$output_dir"
    find "$input_dir" -type f \( -executable -o -name "*.so*" \) | while read -r binary; do
        # check if ELF file
        if file "$binary" | grep -q "ELF"; then
            debug_file="$output_dir/$(basename "$binary").debug"
            objcopy --only-keep-debug "$binary" "$debug_file" || continue
            echo "strip symbols $binary"
            strip --strip-debug --strip-unneeded "$binary" || continue
            objcopy --add-gnu-debuglink="$debug_file" "$binary" || continue
        else
            echo "Skip non-ELF executable: $binary"
        fi
    done
    echo "Debug symbols have been saved to $output_dir"
}
