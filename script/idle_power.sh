#!/bin/bash
sleep 20
echo "ideal123" | sudo -S /home/ideal/shm/read
nvidia-smi -q -d POWER | sed -n '22,22p'
