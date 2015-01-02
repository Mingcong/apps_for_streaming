#!/bin/bash
app=$1
input=$2
result=$3

./parallel_commands "$app $input out1.txt " "$app $input out2.txt" "./system_power.sh" "./gpu_power.sh" >> $result.txt 2>&1
