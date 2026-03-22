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
