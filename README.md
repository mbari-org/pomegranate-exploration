# Exploring pomegranate for whale song unit classification

These are rather terse notes arising from some initial experimentation
with pomegranate. I apply it on some of the whale song unit sequences
being used as part of the training/classification exercises documented
at [`ecoz2-whale`](https://github.com/ecoz2/ecoz2-whale).
For more details you can look at the scripts themselves.
I may be incoporating some of this in a more organized form as part
of the main exercises. Just let me know if you have any questions.

pomegranate documentation looks very good in general, here are
just a couple of initial pointers:

- https://homes.cs.washington.edu/~jmschr/lectures/pomegranate.html
- https://github.com/jmschrei/pomegranate
- https://pomegranate.readthedocs.io/en/latest/

## Main goals:

- explore pomegranate in general
- use it for comparison with and validation of results using ecoz2,
  especially regarding HMMs

## Installing pomegranate

As documented (and confirmed on a collab notebook, and on a linux box)
installing pomegranate is pretty straightforward.
But, due to [some trouble](https://github.com/jmschrei/pomegranate/issues/555#issuecomment-720729698)
trying to install it on my mac, just went for a dockerization strategy
at least for the time being:

    docker build -t pomegranade .

Script `run-pom.sh` is a helper to run any of the scripts in the container with
the usual volume mapping mechanism to work with host files.

    export POME_EXPLORATION_DIR=`pwd`

## Input sequences

They are generated from [`ecoz2-whale/exerc06`](https://github.com/ecoz2/ecoz2-whale/tree/master/exerc06)
using `ecoz2-seq-show`. Option `--pickle` allows to export a set of sequences in a file.

The following calls `ecoz2-seq-show` multiple time to generate all training and test sequences per class:

    cd path/to/ecoz2-whale/exerc06/
    M=256
    dest="../../pomegranate/M${M}/data"
    mkdir -p ${dest}
    for tt in TRAIN TEST; do
      for class in A Bm C E F G2 I3 II; do
        pickle="${dest}/M${M}_${tt}_sequences_$class.pickle"
        ecoz2 seq show --pickle ${pickle} --class-name=$class --codebook-size=${M} --tt=${tt} tt-list.csv
      done
    done

As scripted above, each generated file has this format: `M<M>_<tt>_sequences_<class>.pickle`

- `M`: codebook size
- `tt`: TRAIN or TEST
- `class`: class name

Example: `M32_TRAIN_sequences_A.pickle`

## HMM training

    cd M2048/
    ../run-pom.sh ./hmm_train.sh 3

## HMM classification

    ../run-pom.sh ./hmm_classify.sh 3

            A        Bm         C         E         F        G2        I3        II

    confusion matrix:
    [[101   0   1   0   0   1   0   0]
    [  3 104   0   0   0   0  13   2]
    [  0   0  87   7   9   5   0   2]
    [  0   0   8 116   0  13   0   6]
    [  0   1   1   2  62   2   0   0]
    [  0   0   2   3   0  57   0   0]
    [  1   3   0   0   0   0  59   2]
    [  0   3   5  21   0  14   7 187]]

    classification report:
                  precision    recall  f1-score   support

               A     0.9619    0.9806    0.9712       103
              Bm     0.9369    0.8525    0.8927       122
               C     0.8365    0.7909    0.8131       110
               E     0.7785    0.8112    0.7945       143
               F     0.8732    0.9118    0.8921        68
              G2     0.6196    0.9194    0.7403        62
              I3     0.7468    0.9077    0.8194        65
              II     0.9397    0.7890    0.8578       237

        accuracy                         0.8495       910
       macro avg     0.8367    0.8704    0.8476       910
    weighted avg     0.8635    0.8495    0.8518       910

## 1-Order Markov chain training

    cd M512/
    ../run-pom.sh ./mchain_train.sh 1

## 1-Order Markov chain classification

    ../run-pom.sh ./mchain_classify.sh 1

             A        Bm         C         E         F        G2        I3        II

    confusion matrix:
    [[102   0   1   0   0   0   0   0]
    [  3 108   1   1   1   0   2   6]
    [  0   0  99   0   5   2   1   3]
    [  2   0  28  98   1   9   0   5]
    [  0   0   6   1  61   0   0   0]
    [  0   0  10   3   1  48   0   0]
    [  1  12   0   0   0   0  48   4]
    [  5   4  26  15   3  16   3 165]]

    classification report:
                  precision    recall  f1-score   support

               A     0.9027    0.9903    0.9444       103
              Bm     0.8710    0.8852    0.8780       122
               C     0.5789    0.9000    0.7046       110
               E     0.8305    0.6853    0.7510       143
               F     0.8472    0.8971    0.8714        68
              G2     0.6400    0.7742    0.7007        62
              I3     0.8889    0.7385    0.8067        65
              II     0.9016    0.6962    0.7857       237

        accuracy                         0.8011       910
       macro avg     0.8076    0.8208    0.8053       910
    weighted avg     0.8247    0.8011    0.8029       910
