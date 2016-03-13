#!/bin/bash

scp -i "$1" ubuntu@"$2":/data/q-net-"$3"000 ./checkpoints/q-net-"$4"-"$3"k
scp -i "$1" ubuntu@"$2":/data/q-net-"$3"000.meta ./checkpoints/q-net-"$4"-"$3"k.meta
if [ "$5" != 'nolog' ]
    then 
        scp -i "$1" ubuntu@"$2":~/project/cs231n-project/logfile ./checkpoints/"$4"-"$3"k.log
fi
