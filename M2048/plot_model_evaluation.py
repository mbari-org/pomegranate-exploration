#!/usr/bin/env python3.9

#  Some _very_ preliminary (and incorrect) precision/recall & ROC plotting:
#
#      ../run-pom.sh ./hmm_classify.sh 3 --evaluate-models
#      model evaluation saved to model_evaluation.pickle
#
#  The following on my host (due to already available dependencies):
#
#      python3 plot_model_evaluation.py

import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import precision_recall_curve, roc_curve

# TODO not clear yet about the score

def plot_precision_recall(model_evaluation):
  for class_name, dic in model_evaluation.items():
    y_test = dic['y_test']
    y_score = dic['y_score']

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.plot(recall, precision, lw=2, label=class_name)

  plt.xlabel("recall")
  plt.ylabel("precision")
  plt.legend(loc="best")
  plt.title("precision vs. recall curve")
  plt.show()


def plot_roc(model_evaluation):
  for class_name, dic in model_evaluation.items():
    y_test = dic['y_test']
    y_score = dic['y_score']

    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, lw=2, label=class_name)

  plt.plot([0, 1], [0, 1], 'k--', lw=2)
  plt.xlabel("false positive rate")
  plt.ylabel("true positive rate")
  plt.legend(loc="best")
  plt.title("ROC curve")
  plt.show()


def main():
  filename = 'model_evaluation.pickle'
  with (open(filename, "rb")) as openfile:
    model_evaluation = pickle.load(openfile)

  print('model evaluation loaded {}'.format(filename))
  # print(model_evaluation)

  plot_precision_recall(model_evaluation)
  plot_roc(model_evaluation)


if __name__ == "__main__":
  main()
