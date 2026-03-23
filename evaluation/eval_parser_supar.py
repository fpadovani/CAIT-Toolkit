import os
import sys

# Get absolute path to this file
current_dir = os.path.dirname(os.path.abspath(__file__))

stanza_root = os.path.abspath(os.path.join(current_dir, "..", "stanza"))
sys.path.insert(0, stanza_root)

import stanza
from stanza.utils.conll import CoNLL
from stanza.utils.conll18_ud_eval import load_conllu_file, evaluate

# here you can specify the path to either test or dev set, depending on which one you want to evaluate on
gold_ud = load_conllu_file("UD_English-CHILDES/en_childes-ud-test.conllu")
pred_ud = load_conllu_file("prediction_files/supar_childes_roberta_test.conllu")
scores = evaluate(gold_ud, pred_ud)

print("UAS: {:.2f}".format(100*scores["UAS"].f1))
print("LAS: {:.2f}".format(100*scores["LAS"].f1))
print("MLAS: {:.2f}".format(100*scores["MLAS"].f1))
print("BLEX: {:.2f}".format(100*scores["BLEX"].f1))

# Print total words and sentences for table
print("Total words in gold:", scores["Words"].gold_total)
print("Total words in system:", scores["Words"].system_total)
print("Total sentences in gold:", scores["Sentences"].gold_total)
print("Total sentences in system:", scores["Sentences"].system_total)


## Calculate the LAS and UAS for child speech and child directed speech separately
# Calculate LAS and UAS by sentence length for child speech and child directed speech separately
BASE_GOLD = "UD_English-CHILDES"
BASE_PRED_by_speaker = "prediction_files/prediction_files_by_speaker"

BINS = ["le3", "4to6", "7to10", "gt10"]

GOLD_FILES = {
    "adults":  f"{BASE_GOLD}/en_childes-ud-test_adults",
    "childes": f"{BASE_GOLD}/en_childes-ud-test_childes",
}

PRED_FILES = {
    "Stanza": {
        "adults":  f"{BASE_PRED_by_speaker}/off_the_shelf_test_adults",
        "childes": f"{BASE_PRED_by_speaker}/off_the_shelf_test_childes",
    },
    "SuPar": {
        "adults":  f"{BASE_PRED_by_speaker}/supar_childes_roberta_adults",
        "childes": f"{BASE_PRED_by_speaker}/supar_childes_roberta_childes",
    },
}

for speaker in ["adults", "childes"]:
    print(f"\n{'='*60}")
    print(f"  SPEAKER: {speaker.upper()}")
    print(f"{'='*60}")
    for parser, speaker_files in PRED_FILES.items():
        print(f"\n  Parser: {parser}")
        print(f"  {'-'*40}")
        for bin_label in BINS:
            gold_path = f"{GOLD_FILES[speaker]}_{bin_label}.conllu"
            pred_path = f"{speaker_files[speaker]}_{bin_label}.conllu"
            try:
                gold_ud = load_conllu_file(gold_path)
                pred_ud = load_conllu_file(pred_path)
                scores = evaluate(gold_ud, pred_ud)
                las = 100 * scores["LAS"].f1
                uas = 100 * scores["UAS"].f1
                n_sents = scores["Sentences"].gold_total
                print(f"  [{bin_label:>6}]  LAS: {las:.2f}  UAS: {uas:.2f}  (n={n_sents} sents)")
            except Exception as e:
                print(f"  [{bin_label:>6}]  ERROR: {e}")