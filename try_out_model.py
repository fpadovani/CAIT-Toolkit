from supar import Parser

parser = Parser.load('models/biaffine_roberta_large_childes_10/brlc')
dataset = parser.predict('I saw Sarah with a telescope.', lang='en', prob=True, verbose=False)
sent = dataset[0]

pred_tokens = []
for word, arc, rel in zip(sent.words,sent.arcs, sent.rels): 
    # iterate tokens 
    pred_tokens.append({ "form": word, "upos": '', "xpos": '', "head": arc, "deprel": rel })


print(pred_tokens)