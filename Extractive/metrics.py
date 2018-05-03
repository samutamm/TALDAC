import sys
import os

from rouge import Rouge

def get_rouge_scores(source_summaries, generated_summaries):
    # source: list of source documents
    # generated: list of generated texts
    # return the Rouge-1 Score, Rouge-2 Score, Rouge-l Score
    rouge = Rouge()
    return rouge.get_scores(hyps=source_summaries, refs=generated_summaries, avg=True)['rouge-1']
