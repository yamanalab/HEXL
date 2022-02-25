#!/bin/bash

set -eu

rm -rf build
cmake -S . -B build \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=yes \
        -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build -j
cmake --install build
