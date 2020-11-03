#!/usr/bin/bash
set -ue

order=$1

M=512

for class in A Bm C E F G2 I3 II; do
  mchain_train.py -O=${order} -M=${M} --sequences-filename=data/M${M}_TRAIN_sequences_$class.pickle --class-name=$class
done
