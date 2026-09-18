"""Uni-Sign's translation metric, vendored so numbers match their reported ones exactly.
BLEU: their bundled sacrebleu (external_metrics/sacrebleu.py, tokenizer 13a, corpus-level).
ROUGE-L: the `rouge` pip package, F-measure x100. Copied from SLRT_metrics.py (commit eed438b)."""
from .external_metrics import sacrebleu


def sableu(references, hypotheses, tokenizer="13a"):
    bleu_scores = sacrebleu.corpus_bleu(sys_stream=hypotheses, ref_streams=[references], tokenize=tokenizer).scores
    return {"bleu" + str(n + 1): bleu_scores[n] for n in range(len(bleu_scores))}


def translation_performance(txt_ref, txt_hyp):
    from rouge import Rouge as SLT_Rouge
    scores = SLT_Rouge().get_scores(txt_hyp, txt_ref, avg=True)
    rouge_l = scores["rouge-l"]["f"] * 100
    bleu = sableu(txt_ref, txt_hyp, "13a")
    print(bleu); print(f"Rouge: {rouge_l:.2f}")
    return bleu, float(rouge_l)
