#!/usr/bin/env python3

from pomegranate import *
import numpy as np
from seq import load_sequences
from dist import generate_random_distribution

# NOTE `MarkovChain.from_samples` alone won't work for training when
# not all possible state transitions are present.

def train_model_with_fit(M: int, seqs):
  def initialize():
    # let's initialize with uniform distributions
    p = 1.0 / M
    d1 = DiscreteDistribution({m:p for m in range(M)})
    transitions = []
    for m in range(M):
      for n in range(M):
        transitions.append([m, n, p])

    d2 = ConditionalProbabilityTable(transitions, [d1])

    # https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_6_Markov_Chain.ipynb
    return MarkovChain([d1, d2])

  def train(model):
    model.fit(seqs, inertia=0.1)

  model = initialize()
  train(model)
  return model


def train_model(order: int, M: int, class_name: str, seqs):
  assert(order == 1)
  model = train_model_with_fit(M, seqs)

  # looks like there's no name attribute for a MarkovChain
  return model


def save_model(model, filename):
  with (open(filename, "w")) as openfile:
    openfile.write(model.to_json())
    print('\n{} saved'.format(filename))


def main(args):
  order = int(args.O)
  M = int(args.M)
  class_name = args.class_name
  seqs_filename = args.sequences_filename

  seqs = load_sequences(seqs_filename)
  print('loaded {} sequences'.format(len(seqs)))
  model = train_model(order, M, class_name, seqs)
  # print(model)

  save_model(model, 'data/class_{}_order_{}_mchain.json'.format(class_name, order))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='Training',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-O',  metavar='#', default=1)
    parser.add_argument('-M',  metavar='#')
    parser.add_argument('--class-name', metavar='class')
    parser.add_argument('--sequences-filename')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
  main(parse_args())
