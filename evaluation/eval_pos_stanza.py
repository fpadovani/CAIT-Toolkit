import stanza
from stanza.utils.conll import CoNLL
from stanza.models.pos import scorer

# Load POS-only pipeline
nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,lemma,depparse",
    tokenize_pretokenized=True,
    pos_model_path="stanza_models/pos/en_childes_transformer_tagger.pt"
)

#you can adjust with dev or test set
pred_file = "prediction_files/pos/stanza_pos_childes_charlm_dev.pred.conllu"


gold_doc = CoNLL.conll2doc(
    "UD_English-CHILDES/en_childes-ud-dev.conllu"
)
print(f"Number of sentences in dev set: {len(gold_doc.sentences)}")

# Run POS tagging
pred_doc = nlp(gold_doc)

# Save predictions to file
CoNLL.write_doc2conll(pred_doc, pred_file)

# Evaluate (FILE PATHS, not Documents!)
results = scorer.score(pred_file, 'UD_English-CHILDES/en_childes-ud-dev.conllu')
print(results)