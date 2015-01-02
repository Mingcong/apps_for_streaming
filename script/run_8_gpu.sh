#!/bin/bash
app=$1
input=$2
result=$3

./parallel_commands "$app $input out1.txt " "$app $input out2.txt" "$app $input out3.txt" "$app $input out4.txt" "$app $input out5.txt" "$app $input out6.txt" "$app $input out7.txt" "$app $input out8.txt" "./system_power.sh" "./gpu_power.sh" >> $result.txt 2>&1
