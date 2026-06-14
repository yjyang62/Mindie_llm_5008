#!/bin/bash
cd $CODE_ROOT/src/kernels
if ! ls dist/mie_ops*.whl >/dev/null 2>&1; then
    echo "Wheel packages not found in the directory 'dist'. Start to build mie_ops."
    bash build.sh
fi
if [ ! -d "dist" ]; then
    echo "The directory of mie_ops wheel package 'dist' not found. Skipping wheel installation."
else
    echo "Start to install mie_ops found in the directory of mie_ops wheel package 'dist'."
    find dist -name "mie_ops*.whl" -exec pip install --force-reinstall --target $OUTPUT_DIR/lib {} \;
fi
cd -
