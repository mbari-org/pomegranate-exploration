#!/usr/bin/env python3

from pomegranate import *
import numpy
import pickle
from seq import load_sequences
from dist import generate_random_distribution


# returns State, with its name and unifrom discrete distribution
def generate_state_with_uniform_distribution(state_name, observations):
    temp_dist = DiscreteDistribution(generate_random_distribution(observations))

    temp_state = State(temp_dist, name=state_name)
    return temp_state


def get_uniform_dist_for_all_states(unique_states, unique_syscalls):
    ret_states = []
    for st in unique_states:
        temp_dist = DiscreteDistribution(generate_random_distribution(unique_syscalls))
        temp_state = State(temp_dist, name=st)
        ret_states.append(temp_state)

    return ret_states


def add_uniform_transitions_to_hmm(model, states):
    state_count = len(states)
    dist = 1 / state_count
    for st_from in states:
        for st_to in states:
            model.add_transition(st_from, st_to, dist)
        model.add_transition(model.start, st_from, dist)


def init_model(N: int, M: int, class_name: str):
  print('init_model: N={} M={} class_name={}'.format(N, M, class_name))
  unique_states = ['state{}'.format(n) for n in range(N)]
  unique_syscalls = [m for m in range(M)]
  final_states = get_uniform_dist_for_all_states(unique_states, unique_syscalls)

  model = HiddenMarkovModel(class_name)
  model.add_states(final_states)
  add_uniform_transitions_to_hmm(model, final_states)
  model.bake()
  return model


def fit_model(model, seqs):
  model.fit(sequences=seqs,
    max_iterations=10,
    distribution_inertia=0.7,
    # emission_pseudocount=5,
    edge_inertia=0.25,
    use_pseudocount=True,
    # transition_pseudocount=20,
    # n_jobs=-1,
    verbose=True
  )
  # print(model)


def train_model(N: int, M: int, class_name: str, seqs):
  model = init_model(N, M, class_name)
  fit_model(model, seqs)
  return model


def save_model(model, filename):
  with (open(filename, "w")) as openfile:
    openfile.write(model.to_json())
    print('\n{} saved'.format(filename))


def main(args):
  N = int(args.N)
  M = int(args.M)
  class_name = args.class_name
  seqs_filename = args.sequences_filename

  seqs = load_sequences(seqs_filename)
  print('loaded {} sequences'.format(len(seqs)))
  model = train_model(N, M, class_name, seqs)
  # print(model)
  # print('dense_transition_matrix:')
  # print(model.dense_transition_matrix())

  # for s in seqs:
  #   print('   {}'.format(model.log_probability(s)))

  save_model(model, '{}/N{}_M{}_{}_hmm.json'.format(
    args.dest_dir, N, M, class_name
  ))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='Training',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-N',  metavar='#')
    parser.add_argument('-M',  metavar='#')
    parser.add_argument('--class-name', metavar='class')
    parser.add_argument('--sequences-filename')
    parser.add_argument('--dest-dir')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
  main(parse_args())
