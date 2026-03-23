import os
import sys

# Get absolute path to this file
current_dir = os.path.dirname(os.path.abspath(__file__))

stanza_root = os.path.abspath(os.path.join(current_dir, "..", "stanza"))
sys.path.insert(0, stanza_root)

from stanza.utils.conll import CoNLL
import numpy as np
from scipy.stats import ttest_rel

def sentence_scores(gold_file, pred_file):
    gold_doc = CoNLL.conll2doc(gold_file)
    pred_doc = CoNLL.conll2doc(pred_file)

    uas_scores = []
    las_scores = []

    for gold_sent, pred_sent in zip(gold_doc.sentences, pred_doc.sentences):

        correct_uas = 0
        correct_las = 0
        total = 0

        for gold_word, pred_word in zip(gold_sent.words, pred_sent.words):


            if gold_word.id is None or isinstance(gold_word.id, tuple):
                continue

            total += 1

            if gold_word.head == pred_word.head:
                correct_uas += 1

                if gold_word.deprel == pred_word.deprel:
                    correct_las += 1

        if total > 0:
            uas_scores.append(correct_uas / total)
            las_scores.append(correct_las / total)

    return np.array(uas_scores), np.array(las_scores)



##STANZA off-the-shelf 
uas_a, las_a = sentence_scores("UD_English-CHILDES/en_childes-ud-test.conllu", "prediction_files/off_the_shelf_test_childes.conllu")

#CAIT parser
uas_b, las_b = sentence_scores("UD_English-CHILDES/en_childes-ud-test.conllu", "prediction_files/supar_childes_roberta_test.conllu")

# Paired t-test
t_uas, p_uas = ttest_rel(uas_a, uas_b)
t_las, p_las = ttest_rel(las_a, las_b)

print("UAS p-value:", p_uas)
print("LAS p-value:", p_las)

mean_diff_uas = np.mean(uas_b - uas_a)
mean_diff_las = np.mean(las_b - las_a)

print("Mean UAS difference:", mean_diff_uas)
print("Mean LAS difference:", mean_diff_las)

def cohens_d(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff)

print("Cohen's d (UAS):", cohens_d(uas_b, uas_a))
print("Cohen's d (LAS):", cohens_d(las_b, las_a))
