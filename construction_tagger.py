"""
Construction Tagger for CAIT-Toolkit

Construction-type classification for utterances based on Universal Dependencies (UD) annotations. This can be used with:
1. CoNLL-U files from any UD parser or manually annotated
2. The CAIT stanza+supar pipeline for parsing raw text

Construction types (largely following Cameron-Faulkner, Lieven & Tomasello 2003; Bunzeck & Diessel 2025):
- FOR: Formulaic expressions (greetings, farewells, politeness formulas)
- FRA: Fragments (utterances without a verb)
- QWH: Wh-questions (introduced by interrogative pronouns)
- QYN: Yes/no-questions (auxiliary inversion or declarative + question mark)
- COP: Copula constructions (subject-predicate with copula verb)
- IMP: Imperatives (utterance-initial verb in imperative mood)
- SPI: Subject-predicate intransitive (verb without direct object)
- SPT: Subject-predicate transitive (verb with at least one direct object)
- COM: Complex (multiple independent verbs, sub-/coordination pattern)

Disclaimer: The initial tagging logic was created manually, based on experiences from Bunzeck & Diessel (2025) as well as Bunzeck, Duran & Zarrieß (2025). Claude Code was used for integration with CoNLL-U format and clean-up of code and comments.
"""

import re
import os
import pickle
from typing import List, Optional, Union
from dataclasses import dataclass, field


# =============================================================================
# CoNLL-U Data Structures
# =============================================================================

@dataclass
class ConlluToken:
    """
    Represents a single token in CoNLL-U format.

    CoNLL-U format columns:
    1. ID: Word index (integer starting at 1, or range for MWE, or decimal for empty nodes)
    2. FORM: Word form or punctuation symbol
    3. LEMMA: Lemma or stem
    4. UPOS: Universal POS tag
    5. XPOS: Language-specific POS tag
    6. FEATS: Morphological features
    7. HEAD: Head of current word (0 = root)
    8. DEPREL: Dependency relation to HEAD
    9. DEPS: Enhanced dependency graph
    10. MISC: Any other annotation
    """
    id: int
    form: str
    lemma: str = "_"
    upos: str = "_"
    xpos: str = "_"
    feats: str = "_"
    head: int = 0
    deprel: str = "_"
    deps: str = "_"
    misc: str = "_"

    # Aliases for compatibility with categorization code
    @property
    def text(self) -> str:
        return self.form

    def to_conllu_line(self) -> str:
        """Convert token to a CoNLL-U formatted line."""
        return "\t".join([
            str(self.id),
            self.form,
            self.lemma,
            self.upos,
            self.xpos,
            self.feats if self.feats else "_",
            str(self.head),
            self.deprel,
            self.deps,
            self.misc
        ])

    @classmethod
    def from_conllu_line(cls, line: str) -> Optional["ConlluToken"]:
        """Parse a CoNLL-U line into a ConlluToken."""
        parts = line.strip().split("\t")
        if len(parts) != 10:
            return None

        # Skip multi-word tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
        if "-" in parts[0] or "." in parts[0]:
            return None

        return cls(
            id=int(parts[0]),
            form=parts[1],
            lemma=parts[2],
            upos=parts[3],
            xpos=parts[4],
            feats=parts[5] if parts[5] != "_" else "",
            head=int(parts[6]) if parts[6] != "_" else 0,
            deprel=parts[7],
            deps=parts[8],
            misc=parts[9]
        )


@dataclass
class ConlluSentence:
    """
    Represents a sentence in CoNLL-U format.

    Stores tokens and optional metadata (comments from CoNLL-U file).
    """
    tokens: List[ConlluToken] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Alias for compatibility
    @property
    def words(self) -> List[ConlluToken]:
        return self.tokens

    def to_conllu(self) -> str:
        """Convert sentence to CoNLL-U format string."""
        lines = []

        # Add metadata comments
        for key, value in self.metadata.items():
            lines.append(f"# {key} = {value}")

        # Add token lines
        for token in self.tokens:
            lines.append(token.to_conllu_line())

        return "\n".join(lines)

    def get_text(self) -> str:
        """Reconstruct the sentence text from tokens."""
        if "text" in self.metadata:
            return self.metadata["text"]
        return " ".join(t.form for t in self.tokens if t.upos != "PUNCT") + \
               "".join(t.form for t in self.tokens if t.upos == "PUNCT")

    @classmethod
    def from_conllu_block(cls, block: str) -> "ConlluSentence":
        """Parse a CoNLL-U sentence block (lines separated by newlines)."""
        sentence = cls()

        for line in block.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # Parse metadata comment
                if " = " in line:
                    key, value = line[2:].split(" = ", 1)
                    sentence.metadata[key.strip()] = value.strip()
            else:
                # Parse token
                token = ConlluToken.from_conllu_line(line)
                if token:
                    sentence.tokens.append(token)

        return sentence


def read_conllu_file(filepath: str) -> List[ConlluSentence]:
    """
    Read a CoNLL-U file and return a list of ConlluSentence objects.

    Args:
        filepath: Path to the .conllu file

    Returns:
        List of ConlluSentence objects
    """
    sentences = []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by double newlines (sentence boundaries)
    blocks = content.strip().split("\n\n")

    for block in blocks:
        if block.strip():
            sentence = ConlluSentence.from_conllu_block(block)
            if sentence.tokens:
                sentences.append(sentence)

    return sentences


def write_conllu_file(sentences: List[ConlluSentence], filepath: str) -> None:
    """
    Write a list of ConlluSentence objects to a CoNLL-U file.

    Args:
        sentences: List of ConlluSentence objects
        filepath: Path to output .conllu file
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences):
            f.write(sentence.to_conllu())
            if i < len(sentences) - 1:
                f.write("\n\n")
        f.write("\n")


# =============================================================================
# CAIT Parsing Pipeline (Stanza + SuPar)
# =============================================================================

def get_stanza_pos_pipeline(pos_model_path: Optional[str] = None):
    """
    Initialize Stanza pipeline for tokenization, POS tagging, and lemmatization.

    Args:
        pos_model_path: Optional path to custom POS model (e.g., CHILDES-trained)
    """
    import stanza

    kwargs = {
        "lang": "en",
        "processors": "tokenize,pos,lemma",
    }

    if pos_model_path:
        kwargs["pos_model_path"] = pos_model_path

    return stanza.Pipeline(**kwargs)


def get_supar_parser(model_path: str):
    """
    Initialize the SuPar dependency parser.

    Args:
        model_path: Path to the trained SuPar model
    """
    from supar import Parser
    return Parser.load(model_path)


def parse_utterances(
    utterances: List[str],
    supar_model_path: str,
    stanza_pos_model_path: Optional[str] = None,
    cache_file: Optional[str] = None,
    strip_tags: bool = True,
    verbose: bool = True
) -> List[Optional[ConlluSentence]]:
    """
    Parse utterances using Stanza (POS) + SuPar (dependencies).

    Args:
        utterances: List of utterance strings to parse
        supar_model_path: Path to the trained SuPar model
        stanza_pos_model_path: Optional path to custom Stanza POS model
        cache_file: Optional path to cache parsed results
        strip_tags: Whether to strip tag questions before parsing
        verbose: Whether to print progress

    Returns:
        List of ConlluSentence objects (None for failed parses)
    """
    # Optionally strip tag questions
    if strip_tags:
        utterances_for_parsing = [strip_tag_question(utt) for utt in utterances]
    else:
        utterances_for_parsing = utterances

    # Check cache
    if cache_file and os.path.exists(cache_file):
        if verbose:
            print(f"Loading parsed utterances from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        if cached_data.get("utterances") == utterances_for_parsing:
            return cached_data["parsed"]
        if verbose:
            print("Cache mismatch, re-parsing...")

    if verbose:
        print("Parsing utterances...")
        print("Step 1: Tokenizing and POS tagging with Stanza...")

    stanza_nlp = get_stanza_pos_pipeline(stanza_pos_model_path)

    # First pass: tokenize and POS tag with Stanza
    stanza_results = []
    for i, utt in enumerate(utterances_for_parsing):
        doc = stanza_nlp(utt)
        if doc.sentences:
            stanza_results.append(doc.sentences[0])
        else:
            stanza_results.append(None)
        if verbose and (i + 1) % 500 == 0:
            print(f"  Stanza processed {i + 1}/{len(utterances_for_parsing)}")

    if verbose:
        print("Step 2: Dependency parsing with SuPar...")

    supar_parser = get_supar_parser(supar_model_path)

    # Prepare tokenized inputs for SuPar
    tokenized_inputs = []
    for stanza_sent in stanza_results:
        if stanza_sent is not None:
            tokens = [w.text for w in stanza_sent.words]
            tokenized_inputs.append(" ".join(tokens))
        else:
            tokenized_inputs.append("")

    # Batch parse with SuPar
    supar_results = supar_parser.predict(tokenized_inputs, lang="en", verbose=False)

    if verbose:
        print("Step 3: Combining results into CoNLL-U format...")

    # Combine into ConlluSentence objects
    parsed = []
    for i, (stanza_sent, supar_sent, orig_utt) in enumerate(
        zip(stanza_results, supar_results, utterances_for_parsing)
    ):
        if stanza_sent is None:
            parsed.append(None)
            continue

        stanza_words = stanza_sent.words
        supar_words = supar_sent.words
        supar_arcs = supar_sent.arcs
        supar_rels = supar_sent.rels

        # Check alignment
        if len(stanza_words) != len(supar_words):
            if verbose:
                print(f"Warning: Token mismatch at utterance {i}")
            parsed.append(None)
            continue

        # Create ConlluSentence
        sentence = ConlluSentence()
        sentence.metadata["text"] = orig_utt

        for j, sw in enumerate(stanza_words):
            token = ConlluToken(
                id=sw.id,
                form=sw.text,
                lemma=sw.lemma if sw.lemma else "_",
                upos=sw.upos if sw.upos else "_",
                xpos=sw.xpos if sw.xpos else "_",
                feats=sw.feats if sw.feats else "_",
                head=supar_arcs[j],
                deprel=supar_rels[j],
            )
            sentence.tokens.append(token)

        parsed.append(sentence)

        if verbose and (i + 1) % 500 == 0:
            print(f"  Combined {i + 1}/{len(utterances_for_parsing)}")

    # Cache results
    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump({"utterances": utterances_for_parsing, "parsed": parsed}, f)
        if verbose:
            print(f"Cached parsed utterances to: {cache_file}")

    return parsed


# =============================================================================
# Tag Question Handling
# =============================================================================

TAG_QUESTION_PATTERN = re.compile(
    r",\s*"
    r"(?:"
    r"(?:isn't|aren't|wasn't|weren't|don't|doesn't|didn't|hasn't|haven't|hadn't|can't|couldn't|won't|wouldn't|shouldn't)"
    r"\s+(?:it|he|she|they|we|you|I)"
    r"|"
    r"(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)"
    r"\s+(?:it|he|she|they|we|you|I)"
    r"|"
    r"(?:right|yeah|yes|no|okay|ok|huh|eh)"
    r")"
    r"\s*\??\s*$",
    re.IGNORECASE
)


def strip_tag_question(utterance: str) -> str:
    """
    Remove tag questions from the end of utterances.
    E.g., "that's good isn't it?" -> "that's good."
    """
    stripped = TAG_QUESTION_PATTERN.sub("", utterance)
    stripped = stripped.rstrip(" ,")
    if stripped and stripped[-1] not in ".!?":
        stripped += "."
    return stripped if stripped else utterance


# =============================================================================
# Formulaic Expressions
# =============================================================================

FORMULAIC_PATTERNS = {
    # Greetings
    "hello", "hi", "hey",
    # Farewells
    "bye", "bye bye", "goodbye", "good bye", "good night", "good morning",
    "good afternoon", "good evening", "night night",
    # Politeness
    "thank you", "thanks", "no thanks", "please", "sorry", "excuse me", "pardon",
    # Interjections
    "oops", "oopsydaisy", "oopsy daisy", "whoops", "whoops a daisy", "whoopsadaisy",
    "uh oh", "uh-oh", "oh no", "ouch", "ow", "wow", "hooray", "yay",
    "sshh", "shh", "shush", "hush",
    # Blessings
    "bless you", "gesundheit",
}


def is_formulaic(utterance: str) -> bool:
    """Check if utterance is exactly a formulaic expression."""
    normalized = re.sub(r"[.?!,]+$", "", utterance.lower().strip()).strip()
    return normalized in FORMULAIC_PATTERNS


# =============================================================================
# Construction Type Categorization
# =============================================================================

def categorize_utterance(
    sentence: Optional[ConlluSentence],
    utterance_text: Optional[str] = None
) -> str:
    """
    Categorize a CoNLL-U annotated sentence into a construction type.

    Args:
        sentence: ConlluSentence object with UD annotations
        utterance_text: Optional original utterance text (for formulaic detection)

    Returns:
        Construction type label: FOR, FRA, QWH, QYN, COP, IMP, SPI, SPT, or COM
    """
    if sentence is None:
        return "FRA"

    tokens = sentence.tokens
    deprels = [token.deprel for token in tokens]

    # Get utterance text
    if utterance_text is None:
        utterance_text = sentence.get_text()

    # --- 0. FOR (Formulaic) ---
    if is_formulaic(utterance_text):
        return "FOR"

    # Content tokens (non-punctuation)
    content_tokens = [t for t in tokens if t.upos != "PUNCT"]

    # Find root
    root = next((t for t in tokens if t.head == 0), None)

    # Check for question mark
    has_question_mark = any(t.form == "?" for t in tokens)

    # Helper: identify wh-words
    wh_lemmas = {"who", "what", "where", "when", "why", "how", "which", "whose", "wherever"}
    wh_words = [
        t for t in tokens
        if t.lemma.lower() in wh_lemmas or (t.feats and "PronType=Int" in t.feats)
    ]

    def is_subordinate_wh(wh_word, all_tokens, root_word):
        if wh_word.deprel == "mark":
            return True
        head = next((t for t in all_tokens if t.id == wh_word.head), None)
        if head:
            subordinate_deprels = {"advcl", "ccomp", "acl", "acl:relcl", "relcl"}
            if head.deprel in subordinate_deprels or head.deprel.startswith("acl"):
                return True
        return False

    def get_first_content_word():
        for t in tokens:
            if t.deprel not in ("vocative", "discourse") and t.upos not in ("PUNCT", "INTJ"):
                return t
        return None

    first_content = get_first_content_word()
    has_aux_inversion = first_content and first_content.upos == "AUX"

    has_main_clause_wh = any(
        not is_subordinate_wh(w, tokens, root)
        for w in wh_words
    ) if wh_words else False

    # --- EARLY FRAGMENT DETECTION ---
    # Incomplete copula: "she's", "it's" (contracted be only, no predicate)
    if root and root.upos == "AUX" and not has_question_mark:
        has_subject = any("subj" in t.deprel for t in tokens)
        is_contracted = root.form.lower() in ("'s", "'re", "'m")
        if has_subject and is_contracted and len(content_tokens) == 2:
            return "FRA"

    # Exclamatives: "what a mess" without finite verb
    if wh_words and not has_question_mark:
        first_wh = wh_words[0]
        if first_wh.id == 1 and first_wh.lemma.lower() == "what":
            if len(content_tokens) >= 3:
                second = content_tokens[1] if len(content_tokens) > 1 else None
                if second and second.form.lower() in ("a", "an"):
                    has_finite_verb = any(
                        t.upos == "VERB" and t.feats and "VerbForm=Fin" in t.feats
                        for t in tokens
                    )
                    if not has_finite_verb:
                        return "FRA"

    # --- 1. QYN (yes/no-question) with aux inversion ---
    if has_aux_inversion and has_question_mark:
        return "QYN"

    # --- 2. QWH (wh-question) ---
    if has_main_clause_wh and not has_aux_inversion:
        wh_positions = [w.id for w in wh_words if not is_subordinate_wh(w, tokens, root)]
        aux_positions = [t.id for t in tokens if t.upos == "AUX"]
        if wh_positions and (not aux_positions or min(wh_positions) < min(aux_positions)):
            return "QWH"

    # --- 3. COM (Complex) ---
    complex_clausal_rels = {"ccomp", "advcl", "acl", "acl:relcl", "parataxis"}
    has_complex_clause = any(
        rel in complex_clausal_rels or rel.startswith("acl") for rel in deprels
    )

    has_conjoined_verbs = any(
        t.upos == "VERB" and t.deprel == "conj" and
        any(v.id == t.head and v.upos == "VERB" for v in tokens)
        for t in tokens
    )

    is_lets_construction = (
        root and root.lemma.lower() == "let" and
        any(t.head == root.id and t.deprel == "xcomp" for t in tokens)
    )

    xcomp_count = sum(1 for rel in deprels if rel == "xcomp")
    if is_lets_construction:
        xcomp_count = max(0, xcomp_count - 1)

    if has_complex_clause or has_conjoined_verbs or xcomp_count > 1:
        return "COM"

    # --- 4. QYN for remaining question patterns ---
    if has_question_mark and root:
        if first_content and first_content.upos == "AUX":
            return "QYN"
        if root.upos == "VERB":
            has_subject = any(t.head == root.id and "subj" in t.deprel for t in tokens)
            has_aux = any(t.upos == "AUX" for t in tokens)
            if has_subject and has_aux:
                return "QYN"

    # --- 5. COP (Copula) ---
    has_cop_rel = "cop" in deprels
    has_existential = any(t.deprel == "expl" for t in tokens)
    is_be_root = (
        root and root.upos in ("AUX", "VERB") and
        root.lemma.lower() in ("be", "'s", "'re", "'m")
    )

    if root and root.upos == "VERB":
        has_be_aux = any(
            t.upos == "AUX" and t.head == root.id and
            t.lemma.lower() in ("be", "'s", "'re", "'m", "is", "are", "was", "were", "been")
            for t in tokens
        )
        is_participle = root.feats and "VerbForm=Part" in root.feats
        has_adj_xcomp = any(
            t.deprel == "xcomp" and t.upos == "ADJ" and t.head == root.id
            for t in tokens
        )
        is_passive = any("pass" in t.deprel for t in tokens)
        if has_be_aux and is_participle and (has_adj_xcomp or is_passive):
            return "COP"

    if has_cop_rel or (has_existential and is_be_root):
        return "COP"

    if root and root.upos == "AUX" and root.lemma.lower() in ("be", "'s", "'re", "'m"):
        has_subject = any("subj" in t.deprel for t in tokens)
        if has_subject:
            return "COP"

    # --- 6. FRA (Fragment) ---
    if not root or root.upos not in ("VERB", "AUX"):
        return "FRA"

    if root.upos == "VERB":
        is_participle = root.feats and "VerbForm=Part" in root.feats
        has_aux = any(t.upos == "AUX" and t.head == root.id for t in tokens)
        has_subject = any(t.head == root.id and "subj" in t.deprel for t in tokens)
        if is_participle and not has_aux and not has_subject:
            return "FRA"

    # --- 7. IMP (Imperative) ---
    if root.upos == "VERB":
        is_imp_mood = root.feats and "Mood=Imp" in root.feats
        is_participle = root.feats and "VerbForm=Part" in root.feats
        is_past_tense = root.feats and "Tense=Past" in root.feats
        has_subject = any(t.head == root.id and "subj" in t.deprel for t in tokens)
        has_aux = any(t.upos == "AUX" and t.head == root.id for t in tokens)

        if is_imp_mood:
            return "IMP"

        # "you + verb" imperatives
        if has_subject and not has_aux:
            subj = next((t for t in tokens if t.head == root.id and "subj" in t.deprel), None)
            if subj and subj.lemma.lower() == "you" and subj.id == 1:
                if not is_past_tense and not is_participle:
                    return "IMP"

        # Base verb without subject/aux
        if not has_subject and not has_aux and not is_participle and not is_past_tense:
            return "IMP"

    # --- 8. SPI for AUX root (elliptical responses) ---
    if root and root.upos == "AUX":
        has_subject = any("subj" in t.deprel for t in tokens)
        if has_subject:
            return "SPI"

    # --- 9. SPT (Transitive) ---
    if root and root.upos == "VERB":
        has_direct_obj = any(
            t.deprel == "obj" and (
                t.head == root.id or
                any(v.id == t.head and v.deprel == "xcomp" and v.head == root.id for v in tokens)
            )
            for t in tokens
        )

        control_verbs = {
            "want", "need", "like", "love", "hate", "try", "start", "begin",
            "continue", "prefer", "hope", "wish", "expect", "intend", "plan",
            "decide", "refuse", "agree", "promise", "fail", "manage", "tend",
            "learn", "remember", "forget", "mean", "afford", "deserve"
        }
        has_control_xcomp = (
            root.lemma.lower() in control_verbs and
            any(t.head == root.id and t.deprel == "xcomp" for t in tokens)
        )

        if has_direct_obj or has_control_xcomp:
            return "SPT"

    # --- 10. SPI (Intransitive) ---
    if root and root.upos == "VERB":
        return "SPI"

    return "FRA"


def tag_constructions(
    sentences: List[ConlluSentence],
    utterance_texts: Optional[List[str]] = None
) -> List[str]:
    """
    Tag a list of sentences with construction types.

    Args:
        sentences: List of ConlluSentence objects
        utterance_texts: Optional list of original utterance texts

    Returns:
        List of construction type labels
    """
    if utterance_texts is None:
        utterance_texts = [None] * len(sentences)

    return [
        categorize_utterance(sent, utt)
        for sent, utt in zip(sentences, utterance_texts)
    ]


def tag_conllu_file(
    input_path: str,
    output_path: Optional[str] = None,
    output_format: str = "csv"
) -> List[str]:
    """
    Read a CoNLL-U file, tag constructions, and write output.

    Args:
        input_path: Path to input CoNLL-U file
        output_path: Optional path to output file
        output_format: Output format - "csv" (default) or "conllu"

    Returns:
        List of construction type labels
    """
    sentences = read_conllu_file(input_path)
    tags = tag_constructions(sentences)

    if output_path:
        if output_format == "csv":
            write_results_csv(sentences, tags, output_path)
        elif output_format == "conllu":
            for sent, tag in zip(sentences, tags):
                sent.metadata["construction"] = tag
            write_conllu_file(sentences, output_path)

    return tags


def write_results_csv(
    sentences: List[ConlluSentence],
    tags: List[str],
    output_path: str
) -> None:
    """
    Write tagging results to a CSV file.

    Args:
        sentences: List of ConlluSentence objects
        tags: List of construction type labels
        output_path: Path to output CSV file
    """
    import csv

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "construction"])

        for sent, tag in zip(sentences, tags):
            text = sent.get_text()
            writer.writerow([text, tag])


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Construction tagger for UD-annotated utterances"
    )
    parser.add_argument(
        "input",
        help="Input: CoNLL-U file path, or 'demo' for demonstration"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (CSV by default, or CoNLL-U with --format conllu)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["csv", "conllu"],
        default="csv",
        help="Output format: csv (default) or conllu"
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Parse raw text instead of reading CoNLL-U"
    )
    parser.add_argument(
        "--supar-model",
        default="models/biaffine_roberta_large_childes_10/brlc",
        help="Path to SuPar model (for --parse mode)"
    )
    parser.add_argument(
        "--pos-model",
        help="Path to Stanza POS model (for --parse mode)"
    )

    args = parser.parse_args()

    if args.input == "demo":
        # Demo mode: show examples for each construction type
        demo_sentences = [
            "Hello!",                           # FOR
            "A red ball.",                      # FRA
            "What did you eat?",                # QWH
            "Did you eat the apple?",           # QYN
            "She is happy.",                    # COP
            "Come here!",                       # IMP
            "The dog runs.",                    # SPI
            "She ate the apple.",               # SPT
            "I think he went home.",            # COM
        ]

        print("Demo mode - construction types:")
        print("-" * 50)

        for sent in demo_sentences:
            print(f"  {sent}")

        print("\nTo tag these sentences, use --parse mode with a trained model.")

    else:
        # Read and tag CoNLL-U file
        print(f"Reading: {args.input}")

        # Set default output path if not specified
        output_path = args.output
        if output_path is None:
            base = os.path.splitext(args.input)[0]
            ext = ".csv" if args.format == "csv" else ".tagged.conllu"
            output_path = base + "_constructions" + ext

        tags = tag_conllu_file(args.input, output_path, output_format=args.format)

        # Print summary
        from collections import Counter
        counts = Counter(tags)
        print(f"\nConstruction type distribution ({len(tags)} sentences):")
        for ctype in ["FOR", "FRA", "QWH", "QYN", "COP", "IMP", "SPI", "SPT", "COM"]:
            print(f"  {ctype}: {counts.get(ctype, 0)}")

        print(f"\nOutput written to: {output_path}")
