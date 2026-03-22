import os
import sys

# Get absolute path to this file
current_dir = os.path.dirname(os.path.abspath(__file__))

stanza_root = os.path.abspath(os.path.join(current_dir, "..", "stanza"))
sys.path.insert(0, stanza_root)


from stanza.utils.conll18_ud_eval import load_conllu_file, DEPREL, HEAD

# Add EM and UEM calculation
def calculate_exact_match(gold_ud, pred_ud):
    """Calculate EM and UEM scores"""
    
    # Group words by sentences using sentence spans
    gold_sentences = []
    pred_sentences = []
    
    # Build sentence groups for gold
    current_sent = []
    for word in gold_ud.words:
        if word.is_multiword:
            continue
        current_sent.append(word)
        # Check if this is the last word in a sentence
        is_last = False
        for sent_span in gold_ud.sentences:
            if word.span.end == sent_span.end:
                is_last = True
                break
        if is_last:
            gold_sentences.append(current_sent)
            current_sent = []
    if current_sent:
        gold_sentences.append(current_sent)
    
    # Build sentence groups for predictions
    current_sent = []
    for word in pred_ud.words:
        if word.is_multiword:
            continue
        current_sent.append(word)
        is_last = False
        for sent_span in pred_ud.sentences:
            if word.span.end == sent_span.end:
                is_last = True
                break
        if is_last:
            pred_sentences.append(current_sent)
            current_sent = []
    if current_sent:
        pred_sentences.append(current_sent)
    
    em_correct = 0
    uem_correct = 0
    total_sentences = min(len(gold_sentences), len(pred_sentences))
    
    for gold_words, pred_words in zip(gold_sentences, pred_sentences):
        # Check if sentence lengths match
        if len(gold_words) != len(pred_words):
            continue
        
        # Check UEM (all heads correct)
        # Compare HEAD column values (index in sentence, 0 = root)
        uem_match = all(
            g.columns[HEAD] == p.columns[HEAD]
            for g, p in zip(gold_words, pred_words)
        )
        if uem_match:
            uem_correct += 1
        
        # Check EM (all heads and labels correct)
        em_match = all(
            g.columns[HEAD] == p.columns[HEAD] and g.columns[DEPREL] == p.columns[DEPREL]
            for g, p in zip(gold_words, pred_words)
        )
        if em_match:
            em_correct += 1
    
    em_score = (em_correct / total_sentences * 100) if total_sentences > 0 else 0
    uem_score = (uem_correct / total_sentences * 100) if total_sentences > 0 else 0
    
    return em_score, uem_score, em_correct, uem_correct, total_sentences


gold_ud = load_conllu_file("UD_English-CHILDES/en_childes-ud-test.conllu")
pred_ud = load_conllu_file("prediction_files/supar_childes_roberta_test.conllu")
em, uem, em_correct, uem_correct, total = calculate_exact_match(gold_ud, pred_ud)

print(f"\nEM (Exact Match): {em:.2f}% ({em_correct}/{total})")
print(f"UEM (Unlabeled Exact Match): {uem:.2f}% ({uem_correct}/{total})")