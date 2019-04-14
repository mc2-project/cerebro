#!/bin/bash

# https://gist.github.com/ole1986/d9d6be5218affd41796610a35e3b069c
# code by Ole K.

VERSION="1.0.2"
# Interface connect to out lan
INTERFACE="eth0"
# Interface virtual for incomming traffic
VIRTUAL="ifb0"
# set the direction (1 = outgoing only, 2 = incoming only 3 = both)
DIRECTION=3
# Speed
SPEED_DOWNLOAD="128kbit"
SPEED_UPLOAD="128kbit"
# Delay in milliseconds (0 = no delay)
DELAY=700

function show_usage {
    echo
    echo "Bandwidth Control using TC"
    echo "Version: $VERSION | Author: ole1986"
    echo
    echo "Usage: $1 [-r|--remove] [-i|--incoming] [-o|--outgoing] [-d|--delay=] [--uspeed=] [--dspeed=] <IP>"
    echo
    echo "Arguments:"
    echo "  -r|--remove     : removes all traffic control being set"
    echo "  -i|--incoming   : limit the bandwidth only for incoming packetes"
    echo "  -o|--outgoing   : limit the bandwidth only for outgoing packetes"
    echo "  -d|--delay=700  : define the latency in milliseconds (default: 700)"
    echo "  --uspeed=<speed>: define the upload speed (default: 128kbit)"
    echo "  --dspeed=<speed>: define the download speed (default: 128kbit)"
    echo "  <IP>            : the ip address to limit the traffic for"
    echo
    echo "Changelog:"
    echo "v1.0.2 - make use of the 'tc change' instead of removing"
    echo "v1.0.1 - add uspeed and dspeed to setup limit as argument"
    echo "v1.0.0 - initial version"
}

function remove_tc {
    echo "Unlimit traffic on $INTERFACE (in/out)"
    # clean up outgoing
    [[ $(tc qdisc |grep '^qdisc htb 1:') ]] && tc qdisc del root dev $INTERFACE
    # clean up incoming
    if [[ $(tc qdisc | grep '^qdisc htb 2:') ]]; then
        tc qdisc del dev $INTERFACE handle ffff: ingress
        tc qdisc del root dev $VIRTUAL
        # Unload the virtual network module
        ip link set dev $VIRTUAL down
        modprobe -r ifb
    fi
}

function tc_outgoing {
    echo "Limit outgoing traffic"

    # update outgoing speed (if already exists)
    if [[ $(tc class show dev $INTERFACE |grep '^class htb 1:1') ]]; then
        TCC="tc class change"
    else
        # Add classes per ip
        tc qdisc add dev $INTERFACE root handle 1: htb
        TCC="tc class add"
    fi

    echo "- upload speed $SPEED_UPLOAD"
    $TCC dev $INTERFACE parent 1: classid 1:1 htb rate $SPEED_UPLOAD
    if [ $DELAY -gt 0 ]; then
        TCN="tc qdisc add"
        [[ $(tc qdisc |grep '^qdisc netem 10:') ]] && TCN="tc qdisc change"
        echo "- latency ${DELAY}ms"
        $TCN dev $INTERFACE parent 1:1 handle 10: netem delay ${DELAY}ms
    fi

    # Match ip and put it into the respective class
    echo "- filter on IP $ADDRESS"
    TCF="tc filter add"
    if [[ $(tc filter show dev $INTERFACE | grep '^filter parent 1: protocol ip pref 1') ]]; then
        TCF="tc filter change"
    fi
    $TCF dev $INTERFACE protocol ip parent 1: prio 1 u32 match ip dst $ADDRESS flowid 1:1
}

function tc_incoming {
    # setup virtual interface to redirect incoming traffic
    modprobe ifb numifbs=1
    ip link set dev $VIRTUAL up

    echo "Limit incoming traffic"

    # update incoming speed (if already exists)
    if [[ $(tc class show dev $VIRTUAL |grep '^class htb 2:1') ]]; then
        TCC="tc class change"
    else
        tc qdisc add dev $INTERFACE handle ffff: ingress
        # Redirecto ingress eth0 to egress ifb0
        tc filter add dev $INTERFACE parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev $VIRTUAL
        # Add classes per ip
        tc qdisc add dev $VIRTUAL root handle 2: htb

        TCC="tc class add"
    fi

    echo "- download speed $SPEED_DOWNLOAD"
    $TCC dev $VIRTUAL parent 2: classid 2:1 htb rate $SPEED_DOWNLOAD

    if [ $DELAY -gt 0 ]; then
        TCN="tc qdisc add"
        [[ $(tc qdisc |grep '^qdisc netem 20:') ]] && TCN="tc qdisc change"

        echo "- latency ${DELAY}ms"
        $TCN dev $VIRTUAL parent 2:1 handle 20: netem delay ${DELAY}ms
    fi

    echo "- filter on IP $ADDRESS"
    TCF="tc filter add"
    if [[ $(tc filter show dev $VIRTUAL | grep '^filter parent 1: protocol ip pref 1') ]]; then
        TCF="tc filter change"
    fi

    $TCF dev $VIRTUAL protocol ip parent 2: prio 1 u32 match ip src $ADDRESS flowid 2:1
}

if [ $# -eq 0 ]; then
    show_usage $0
    exit 0
fi

for i in "$@"
do
case $i in
    --remove|-r)
    DIRECTION=0
    shift # past argument with no value
    ;;
    --outgoing|-o)
    DIRECTION=1
    shift
    ;;
    --incoming|-i)
    DIRECTION=2
    shift
    ;;
    -d=*|--delay=*)
    DELAY=${i#*=}
    shift
    ;;
    --uspeed=*)
    SPEED_UPLOAD="${i#*=}"
    shift
    ;;
    --dspeed=*)
    SPEED_DOWNLOAD="${i#*=}"
    shift
    ;;
esac
done

remove_tc

[ -n $1 ] && ADDRESS=$1

[ $DIRECTION -eq 0 ] && exit 0

if [ -z $ADDRESS ]; then
    echo
    echo "No IP address defined"
    exit 0
fi

[ $DIRECTION -eq 1 ] && tc_outgoing
[ $DIRECTION -eq 2 ] && tc_incoming

if [ $DIRECTION -eq 3 ]; then
    tc_outgoing
    tc_incoming
fi
