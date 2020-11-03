#!/usr/bin/bash
set -ue

order=$1

M=512

mchain_classify.py \
    --models data/*_order_${order}_mchain.json \
    --sequences-filenames data/M${M}_TEST_sequences_*.pickle
