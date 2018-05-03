import random
import loadData as ld
#Rouge code Github : https://github.com/pltrdy/rouge
from rouge import Rouge


def get_rouge_scores(source_summaries, generated_summaries):
    # source: list of source documents
    # generated: list of generated texts
    # return the Rouge-1 Score, Rouge-2 Score, Rouge-l Score
    rouge = Rouge()
    return rouge.get_scores(hyps=source_summaries, refs=generated_summaries, avg=True)

def evaluate_supervised_model(model, source, source_summaries, N, idxToWords):
    # source: list of source documents
    # source_summaries: list of source summaries
    # model: Seq2Seq Keras model
    # N: Pick n random documents from sources
    # idxToWords: Is like an array where an index corresponds to his word
    # Evaluate a supervised model with ROUGE metrics on N random documents from sources
    
    result_Y = []
    result_summary = []

    random_indexes = random.sample(range(len(source)), N)
    rouge = Rouge()
    
    i=0
    for index in random_indexes:
        if( i%100 == 0):
            print(i)

        result_Y.append(ld.convertIndexToWords(source_summaries[index], idxToWords).replace('_start_', ''))
        summary = model.demo_model_predictions(source[index])
        result_summary.append(summary.replace('_start_', '')) 

        i+=1
    
    print("Done !")

    return rouge.get_scores(hyps=result_Y, refs=result_summary, avg=True)