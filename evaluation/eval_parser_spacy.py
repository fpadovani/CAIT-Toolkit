import os
import sys

# Get absolute path to this file
current_dir = os.path.dirname(os.path.abspath(__file__))

stanza_root = os.path.abspath(os.path.join(current_dir, "..", "stanza"))
sys.path.insert(0, stanza_root)

import stanza
from stanza.utils.conll import CoNLL
from stanza.utils.conll18_ud_eval import load_conllu_file, evaluate


## test or dev files (change accordingly))
gold_ud = load_conllu_file("UD_English-CHILDES/en_childes-ud-test.conllu")
pred_ud = load_conllu_file("prediction_files/spacy_trf_childes_test_mapped.conllu")
scores = evaluate(gold_ud, pred_ud)

print("UAS test: {:.2f}".format(100*scores["UAS"].f1))
print("LAS test: {:.2f}".format(100*scores["LAS"].f1))
print("MLAS test: {:.2f}".format(100*scores["MLAS"].f1))
print("BLEX test: {:.2f}".format(100*scores["BLEX"].f1))


