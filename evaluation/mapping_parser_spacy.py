import spacy
from spacy_conll import ConllFormatter
from pathlib import Path
from spacy.tokens import Doc
from collections import Counter


SPACY_TO_STANZA = {
    "ROOT": "root",                  # root
    "acl": "acl",                    # clausal modifier of noun
    "acomp": "amod",                 # adjectival complement → amod
    "advcl": "advcl",                # adverbial clause modifier
    "advmod": "advmod",              # adverbial modifier
    "agent": "obl",            # agent → oblique agent
    "amod": "amod",                  # adjectival modifier
    "appos": "appos",                # appositional modifier
    "attr": "cop",                 # attribute → closest is open clausal complement
    "aux": "aux",                     # auxiliary
    "auxpass": "aux:pass",            # auxiliary passive
    "case": "case",                   # case marking
    "cc": "cc",                       # coordinating conjunction
    "ccomp": "ccomp",                 # clausal complement
    "compound": "compound",           # compound
    "conj": "conj",                   # conjunct
    "csubj": "csubj",                 # clausal subject
    "csubjpass": "csubj:pass",        # clausal passive subject
    "dative": "iobj",                 # dative → indirect object
    "dep": "dep",                     # unspecified dependency
    "det": "det",                     # determiner
    "dobj": "obj",                    # direct object
    "expl": "expl",                   # expletive
    "intj": "intj",                   # interjection
    "mark": "mark",                   # marker
    "meta": "dep",                     # meta modifier → generic dep
    "neg": "advmod",                  # negation modifier → UD advmod (neg attached via feature in UD)
    "nmod": "nmod",                   # modifier of nominal
    "npadvmod": "advmod",             # noun phrase as adverbial modifier → advmod
    "nsubj": "nsubj",                 # nominal subject
    "nsubjpass": "nsubj:pass",        # passive nominal subject
    "nummod": "nummod",               # numeric modifier
    "oprd": "xcomp",                   # object predicate → closest xcomp
    "parataxis": "parataxis",         # parataxis
    "pcomp": "obl",                   # complement of preposition → oblique
    "pobj": "obj",                    # object of preposition → object
    "poss": "nmod:poss",              # possession modifier → possessive nominal
    "preconj": "cc:preconj",          # pre-correlative conjunction → Stanza subtype
    "predet": "det:predet",           # predet → no direct UD equivalent → generic dep
    "prep": "case",                   # prepositional modifier → UD attaches as case to noun
    "prt": "compound:prt",            # particle → phrasal verb particle
    "punct": "punct",                 # punctuation
    "quantmod": "nummod",             # modifier of quantifier → numeric modifier
    "relcl": "acl:relcl",             # relative clause modifier
    "xcomp": "xcomp"                  # open clausal complement
}


def fix_attr_copula_heads(spacy_conll, stanza_labels):
    """
    If original DEPREL was 'attr' and mapped to 'cop',
    reassign HEAD to the first NOUN to the right.
    """
    n = len(spacy_conll)

    for i, tok in enumerate(spacy_conll):
        if tok["DEPREL"] == "attr" and stanza_labels[i] == "cop":
            new_head = None

            # search rightward for first NOUN
            for j in range(i + 1, n):
                if spacy_conll[j]["UPOS"] == "NOUN":
                    new_head = spacy_conll[j]["ID"]
                    break

            # fallback: attach to root
            if new_head is None:
                for t in spacy_conll:
                    if t["HEAD"] == 0:
                        new_head = t["ID"]
                        break

            tok["HEAD"] = new_head


def renumber_and_fix_duplicates(tokens):
    """
    Renumber tokens only if duplicate IDs exist.
    For second and later occurrences of the same ID:
      - set DEPREL = "attr"
      - set HEAD = 1
    """
    seen = {}
    new_tokens = []

    for i, tok in enumerate(tokens, start=1):
        tok = tok.copy()
        old_id = tok["ID"]

        if old_id in seen:
            # duplicate ID → force repair
            tok["DEPREL"] = "attr"
            tok["HEAD"] = 1
        else:
            seen[old_id] = True

        tok["ID"] = i
        new_tokens.append(tok)

    return new_tokens

def enforce_single_root(spacy_conll):
    roots = [tok for tok in spacy_conll if tok["HEAD"] == 0]

    if len(roots) <= 1:
        return spacy_conll

    main_root = roots[0]["ID"]

    for tok in roots[1:]:
        tok["HEAD"] = main_root
        tok["DEPREL"] = "parataxis"

    return spacy_conll


def renumber_and_fix_duplicates(tokens):
    """
    Renumber tokens if duplicate IDs exist.
    For duplicate IDs:
      - assign DEPREL = "attr"
      - assign HEAD = root token ID
    """

    # --- find root (first token with HEAD == 0) ---
    root_old_id = None
    for tok in tokens:
        if tok["HEAD"] == 0:
            root_old_id = tok["ID"]
            break

    if root_old_id is None:
        # fallback: attach to first token if no root exists
        root_old_id = tokens[0]["ID"]

    seen = {}
    new_tokens = []
    old_to_new = {}

    # --- renumber IDs ---
    for i, tok in enumerate(tokens, start=1):
        tok = tok.copy()
        old_to_new[tok["ID"]] = i
        tok["ID"] = i
        new_tokens.append(tok)

    root_new_id = old_to_new[root_old_id]

    # --- repair duplicates ---
    seen.clear()
    for tok in new_tokens:
        original_id = tok["ID"]

        if original_id in seen:
            tok["DEPREL"] = "attr"
            tok["HEAD"] = root_new_id
        else:
            seen[original_id] = True

    return new_tokens


def has_duplicate_ids(tokens):
    ids = [tok["ID"] for tok in tokens]
    return any(count > 1 for count in Counter(ids).values())


def extract_text_from_metadata(meta_lines):
    for line in meta_lines:
        if line.startswith("# text ="):
            return line.split("=", 1)[1].strip()
    return None


def parse_conllu_sentences(conllu_file):
    sentences = []
    meta_lines = []
    sent_items = []

    with open(conllu_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("#"):
                meta_lines.append(line)

            elif line == "":
                if meta_lines or sent_items:
                    sentences.append((meta_lines, sent_items))
                meta_lines = []
                sent_items = []

            else:
                fields = line.split("\t")
                tok_id = fields[0]

                if "-" in tok_id:
                    sent_items.append({"type": "mwt", "line": line})
                elif "." in tok_id:
                    sent_items.append({"type": "empty", "line": line})
                else:
                    sent_items.append({
                        "type": "tok",
                        "id": int(tok_id),
                        "form": fields[1]
                    })

        if meta_lines or sent_items:
            sentences.append((meta_lines, sent_items))

    return sentences



def predict_spacy_sentence_split(sentence, out_f):
    meta_lines, sent_items = sentence

    # --- write metadata verbatim ---
    for line in meta_lines:
        out_f.write(line + "\n")

    # --- extract gold tokens ---
    real_tokens = [x for x in sent_items if x["type"] == "tok"]
    tokens = [t["form"] for t in real_tokens]

    if not tokens:
        out_f.write("\n")
        return

    doc = Doc(nlp.vocab, words=tokens)

    # --- run spaCy pipeline EXCEPT tokenizer ---
    for name, pipe in nlp.pipeline:
        doc = pipe(doc)

    spacy_conll = [tok for sent in doc._.conll for tok in sent]

    if has_duplicate_ids(spacy_conll):
        spacy_conll = renumber_and_fix_duplicates(spacy_conll)
        spacy_conll = enforce_single_root(spacy_conll)

    mapped_labels = [
        SPACY_TO_STANZA.get(tok["DEPREL"], "dep")
        for tok in spacy_conll
    ]

    fix_attr_copula_heads(spacy_conll, mapped_labels)

    spacy_idx = 0

    for item in sent_items:
        if item["type"] in {"mwt", "empty"}:
            out_f.write(item["line"] + "\n")
            continue

        tok = spacy_conll[spacy_idx]


        # --- Map spaCy DEPREL to Stanza UD v2 ---
        ud_label = SPACY_TO_STANZA.get(tok['DEPREL'], "dep")

        out_f.write(
            f"{tok['ID']}\t{tok['FORM']}\t{tok['LEMMA']}\t"
            f"{tok['UPOS']}\t{tok['XPOS']}\t_\t"
            f"{tok['HEAD']}\t{ud_label}\t"
            f"{tok['HEAD']}:{ud_label}\t"
            f"{tok['MISC']}\n"
        )

        spacy_idx += 1

    out_f.write("\n")


############################################
# Run
############################################


nlp = spacy.load("en_core_web_trf")

nlp.add_pipe("conll_formatter", last=True)


## dev set mapping
gold_conllu_dev = "UD_English-CHILDES/en_childes-ud-dev.conllu"
out_conllu_dev = "prediction_files/spacy_trf_childes_dev_mapped.conllu"

## test set mapping
gold_conllu_test = "UD_English-CHILDES/en_childes-ud-test.conllu"
out_conllu_test = "prediction_files/spacy_trf_childes_test_mapped.conllu"

sentences_dev = parse_conllu_sentences(gold_conllu_dev)
sentences_test = parse_conllu_sentences(gold_conllu_test)

with open(out_conllu_dev, "w", encoding="utf-8") as out_f:
    for sentence in sentences_dev:
        predict_spacy_sentence_split(sentence, out_f)


with open(out_conllu_test, "w", encoding="utf-8") as out_f:
    for sentence in sentences_test:
        predict_spacy_sentence_split(sentence, out_f)


## You can then evaluate the mapped output files using the same evaluation code as for SuPar, 
# just changing the prediction file paths to point to the new mapped files.