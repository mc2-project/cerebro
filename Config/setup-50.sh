#!/bin/bash
sudo ./traffic-control.sh -r
sudo ./traffic-control.sh --uspeed=$1 --dspeed=$2 -d=50 172.31.0.0/16
# sudo tc qdisc add dev eth0 root handle 1:0 tbf rate $1 buffer $3 latency $4
# sudo tc qdisc add dev eth0 parent 1:0 handle 10: netem delay $2
sudo tc filter show dev eth0
sudo tc qdisc show dev eth0
