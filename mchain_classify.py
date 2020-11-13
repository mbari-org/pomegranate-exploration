#!/usr/bin/env python3

from pomegranate import *
import numpy as np
import pickle
import re
from sklearn import metrics
from seq import load_sequences


def load_model(filename: str, args):
  if args.verbose:
    print('loading model {}'.format(filename))

  with (open(filename, "r")) as openfile:
      return MarkovChain.from_json(openfile.read())


def print_model_names(model_names):
  formatted = ['{:>10}'.format(name) for name in model_names]
  print(''.join(formatted))


def extract_class_name_from_sequence_filename(sequences_filename: str) -> str:
  match = re.search(r'sequences_([^/]+)\.pickle$', sequences_filename)
  return match.group(1)


def extract_class_name_from_model_filename(model_filename: str) -> str:
  match = re.search(r'class_([^_]+)_', model_filename)
  return match.group(1)


def main(args):
  # load all individual sequences
  y_true = []
  y_seqs = []
  prob_rows = []  # initialized with dummy prob array per sequence
  for sequences_filename in args.sequences_filenames:
    seq_class_name = extract_class_name_from_sequence_filename(sequences_filename)
    seqs = load_sequences(sequences_filename, args.verbose)
    for seq in seqs:
      y_true.append(seq_class_name)
      y_seqs.append(seq)
      prob_rows.append([0 for _ in args.models])

  print('loaded {} sequences'.format(len(y_seqs)))

  # matrix of probabilities
  prob_rows_index = 0

  # with one model loaded at a time (for memory constraint reasons),
  # get the probability for every loaded sequence:
  for model_index, fn in enumerate(args.models):
    model_name = extract_class_name_from_model_filename(fn)
    print('evaluating probabilities with model {}'.format(model_name))
    model = load_model(fn, args)

    for seq_class_name, seq, prob_row in zip(y_true, y_seqs, prob_rows):
      prob = model.log_probability(seq)
      prob_row[model_index] = prob

  # get the predictions:
  model_names = [extract_class_name_from_model_filename(fn) for fn in args.models]
  y_pred = []
  for probs in prob_rows:
    max_m = np.argmax(probs)
    y_pred.append(model_names[max_m])

  print('\nconfusion matrix:')
  print(metrics.confusion_matrix(y_true, y_pred))

  print('\nclassification report:')
  print(metrics.classification_report(y_true, y_pred, digits=4))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='Classification',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--models', nargs='+')
    parser.add_argument('--sequences-filenames', nargs='+')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
  main(parse_args())
