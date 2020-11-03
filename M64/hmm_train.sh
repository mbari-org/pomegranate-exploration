#!/usr/bin/bash
set -ue

N=$1

M=64

for class in A Bm C E F G2 I3 II; do
  hmm_train.py -N=${N} -M=${M} --sequences-filename=data/M${M}_TRAIN_sequences_$class.pickle --class-name=$class
done
