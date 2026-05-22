#!/bin/bash
cd $CODE_ROOT/src/kernels
bash build.sh
pip install dist/mie_ops*.whl --force-reinstall --target $OUTPUT_DIR/lib
cd -