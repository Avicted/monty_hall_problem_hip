#!/bin/bash

set -xe

[ -d ./build ] && rm -rf ./build
mkdir -p ./build

CC=hipcc
CFLAGS="-O3 -Wall -Wextra -Werror"
LDLIBS="-lm"

SRC_FILES=(
  "./src/main.hip"
)
INCLUDE_DIRS="-I/opt/rocm/include"
OUTPUT_DIR="./build"
EXECUTABLE_NAME="monty_hall_problem_hip.out"


$CC $CFLAGS $INCLUDE_DIRS -o $OUTPUT_DIR/$EXECUTABLE_NAME ${SRC_FILES[@]} $LDLIBS
