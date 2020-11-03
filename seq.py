import pickle

def load_sequences(filename: str, verbose=False):
  if verbose:
      print('loading sequences {}'.format(filename))
  with (open(filename, "rb")) as openfile:
      return pickle.load(openfile)
