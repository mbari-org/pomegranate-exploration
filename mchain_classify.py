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
  models = [load_model(fn, args) for fn in args.models]
  model_names = [extract_class_name_from_model_filename(fn) for fn in args.models]

  print_model_names(model_names)

  y_true = []
  y_pred = []

  for sequences_filename in args.sequences_filenames:
    class_name = extract_class_name_from_sequence_filename(sequences_filename)

    seqs = load_sequences(sequences_filename, args.verbose)

    if args.verbose:
      print('\n{} ({} sequences):'.format(sequences_filename, len(seqs)))

    for seq in seqs:
      # print('seq={}'.format(seq))
      probs = [m.log_probability(seq) for m in models]
      max_m = np.argmax(probs)

      y_true.append(class_name)
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
