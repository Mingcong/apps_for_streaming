#!/bin/bash
app=$1
input=$2
result=$3

for numProcesses in 1 2 4 6 8
do
	echo "app is $app, input is $input and using $numProcesses processes"
	echo "app is $app, input is $input and using $numProcesses processes" >> $result.txt
	program="./run_"$numProcesses"_gpu.sh"
	./parallel_commands "$program $app $input $result" "./gpu_power.sh" >> $result.txt 2>&1
	echo "	" >> $result.txt
	./idle_power.sh >> $result.txt 2>&1
	echo "	" >> $result.txt
done
	
