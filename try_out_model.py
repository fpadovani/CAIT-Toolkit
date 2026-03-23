from supar import Parser
import stanza

pos_tagger_model = 'stanza_models/pos/en_childes_charlm_tagger.pt'
parser = Parser.load('models/biaffine_roberta_large_childes_10/brlc')
tagger = stanza.Pipeline(
        lang='en',
        processors='tokenize,pos,lemma,depparse',
        use_gpu=True,
        pos_model_path=pos_tagger_model)

stanza_doc = tagger('I saw Sarah with a telescope.')
stanza_sent = stanza_doc.sentences[0]
dataset = parser.predict('I saw Sarah with a telescope.', lang='en', prob=True, verbose=False)
sent = dataset[0]

# Combine predictions
pred_tokens = []
for word, arc, rel, stanza_word in zip(sent.words, sent.arcs, sent.rels, stanza_sent.words):
    pred_tokens.append({
        "form": word,
        "upos": stanza_word.upos,    
        "xpos": stanza_word.xpos,    
        "head": arc,
        "deprel": rel
    })

print(pred_tokens)