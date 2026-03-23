import os
import sys

# Get absolute path to this file
current_dir = os.path.dirname(os.path.abspath(__file__))

stanza_root = os.path.abspath(os.path.join(current_dir, "..", "stanza"))
sys.path.insert(0, stanza_root)

import stanza
from stanza.utils.conll import CoNLL
from stanza.utils.conll18_ud_eval import load_conllu_file, evaluate


# Load English pipeline (it loads the Combined off-the-shelf model by default)
nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,lemma,depparse",
    tokenize_pretokenized=True,
)


# Load UD-CHILDES test or dev set
doc = CoNLL.conll2doc("UD_English-CHILDES/en_childes-ud-test.conllu")

doc = nlp(doc)

# Save predictions
CoNLL.write_doc2conll(doc, "prediction_files/off_the_shelf_test_childes.conllu")


gold_ud = load_conllu_file("UD_English-CHILDES/en_childes-ud-test.conllu")
pred_ud = load_conllu_file("prediction_files/off_the_shelf_test_childes.conllu")
scores = evaluate(gold_ud, pred_ud)

print("UAS: {:.2f}".format(100*scores["UAS"].f1))
print("LAS: {:.2f}".format(100*scores["LAS"].f1))
print("MLAS: {:.2f}".format(100*scores["MLAS"].f1))
print("BLEX: {:.2f}".format(100*scores["BLEX"].f1))