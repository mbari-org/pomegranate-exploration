#!/usr/bin/bash
set -ue

N=$1

M=128

hmm_classify.py \
    --models data/N${N}_*_hmm.json \
    --sequences-filenames data/M${M}_TEST_sequences_*.pickle