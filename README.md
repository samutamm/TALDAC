
### Summarization

Dataset cnn_stories from https://cs.nyu.edu/~kcho/DMQA/

Execute run_rouge.py to calcul the rouge score between machine made summaries and human made summaries.

```
./run_rouge.py data/my_generated_resume.csv
```
where file my_generated_resume.csv should contain (at least) two columns:

| Column        | Content           |
| ------------- |:-------------:|
| Y      | Human made summaries. In case of cnn dataset, all highlights concatenated to one bigger summary. |
| summary      | Machine made summaries to be evaluated.     |

The script will print the average precision, recall and f_score.