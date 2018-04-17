
from PyRouge.PyRouge.pyrouge import Rouge
import pandas as pd
import sys

r = Rouge()
if len(sys.argv) < 2:
    print("Please give the path to csv file as a argument.")
    sys.exit(0)
    
stories = pd.read_csv(sys.argv[1]).sample(4)
cols = ['Y','summary']
for col in cols:
    if col not in stories.columns:
        print("csv file is missing column " + col)
        print("Please provide file with columns " + str(cols))
        sys.exit(0)

def calcul_rouge(columns):
    references = columns[0]
    summaries = columns[1]
    return r.rouge_l([summaries], [references])

result = stories[['Y','summary']].apply(calcul_rouge, axis=1)
result = result.apply(pd.Series)
result.columns = ['precision','recall','f_score']
print(result.mean())