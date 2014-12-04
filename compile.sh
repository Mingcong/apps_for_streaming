#!/bin/bash
source_code=$1
echo g++ "$source_code" -o app
g++ "$source_code" -o app `pkg-config --cflags --libs opencv` -L/home/ideal/tools/cuda-5.5/lib64