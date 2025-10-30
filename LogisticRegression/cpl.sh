#!/bin/bash

BUILD_DIR="./bin"
echo "compiling... into ${BUILD_DIR}"

SDK="$(xcrun --sdk macosx --show-sdk-path)"
clang++ -std=c++20 -O3 -stdlib=libc++ \
  -I"$SDK/usr/include/c++/v1" -I"$SDK/usr/include" \
  softmax_sgd_test.cpp -o ${BUILD_DIR}/softmax_sgd_test

clang++ -std=c++20 -O3 -stdlib=libc++ -I"$SDK/usr/include/c++/v1" -I"$SDK/usr/include" trainer_main.cpp -o ${BUILD_DIR}/trainer_main
