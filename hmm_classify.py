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
      return HiddenMarkovModel.from_json(openfile.read())


def print_model_names(models):
  formatted = ['{:>10}'.format(m.name) for m in models]
  print(''.join(formatted))


def extract_class_name_from_sequence_filename(sequences_filename: str) -> str:
  match = re.search(r'sequences_([^/]+)\.pickle$', sequences_filename)
  return match.group(1)


# TODO not clear yet about the score
def evaluate_models(models, args):
  # load all sequences index by class_name:
  seqs_by_class_name = {}
  for sequences_filename in args.sequences_filenames:
    class_name = extract_class_name_from_sequence_filename(sequences_filename)
    seqs = load_sequences(sequences_filename, args.verbose)
    seqs_by_class_name[class_name] = seqs

  # by class name:
  model_evaluation = {}

  # evaluate each model:
  for model_index, model in enumerate(models):
    y_test = []
    y_score = []

    for seq_class_name, seqs in seqs_by_class_name.items():
      for seq in seqs:
        log_probs = [m.probability(seq) for m in models]
        winner_index = np.argmax(log_probs)
        if seq_class_name == model.name:
          y_test.append(int(model_index == winner_index))
        else:
          y_test.append(int(model_index != winner_index))

        score = log_probs[winner_index]

        # sorted_probs = np.sort(log_probs)
        # score = sorted_probs[0] - sorted_probs[1]

        y_score.append(score)

    model_evaluation[model.name] = dict(
      y_test=y_test,
      y_score=y_score,
    )

  return model_evaluation


def main(args):
  models = [load_model(fn, args) for fn in args.models]

  if args.evaluate_models:
    model_evaluation = evaluate_models(models, args)
    # print(model_evaluation)

    filename = 'model_evaluation.pickle'
    with (open(filename, "wb")) as openfile:
      pickle.dump(model_evaluation, openfile)
      print('model evaluation saved to {}'.format(filename))

    return

  print_model_names(models)

  y_true = []
  y_pred = []

  for sequences_filename in args.sequences_filenames:
    class_name = extract_class_name_from_sequence_filename(sequences_filename)

    seqs = load_sequences(sequences_filename, args.verbose)

    if args.verbose:
      print('\n{} ({} sequences):'.format(sequences_filename, len(seqs)))

    for seq in seqs:
      probs = [m.log_probability(seq) for m in models]
      max_m = np.argmax(probs)

      y_true.append(class_name)
      y_pred.append(models[max_m].name)

  print('\nconfusion matrix:')
  print(metrics.confusion_matrix(y_true, y_pred))

  print('\nclassification report:')
  print(metrics.classification_report(y_true, y_pred, digits=4))

  print('MCC = {}'.format(metrics.matthews_corrcoef(y_true, y_pred)))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='Classification',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--models', nargs='+')
    parser.add_argument('--sequences-filenames', nargs='+')
    parser.add_argument('--evaluate-models', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
  main(parse_args())
