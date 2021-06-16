# CS5740 A3

In this assignment, we present 3 generative approaches (Greedy, Beam and Viterbi Search) to predict parts-of-speech (POS) tags given about a million words of text from the Wall Street Journal. POS tags from the [Penn Treebank](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) project are used in these experiments.

F1-score is used to measure the accuracies of predicted tags on the development and test dataset.

## Execution Instructions

To setup the environment for this project, use Python 3.8.x and install all requirements.
```
pip install -r requirements.txt
```

Run the following command to train the trigram model and generate the test results in test_y.csv.
```
python pos_tagger.py
```

## Results

The results reported on the development and test set from these experiments are:

Dataset | System | F1-score |
| ------- | --------- | -------|
| Development Data | 3-gram, Beam Search| 0.9596 |
| Development Data | 3-gram, Viterbi Search| 0.9594|
| Test Data | 3-gram, Beam Search| 0.9616 |
| Test Data | 3-gram, Viterbi Search| 0.9613 |
