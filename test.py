import argparse
from evaluation.rouge_evaluator import RougeEvaluator
import json
import tqdm

args = argparse.ArgumentParser()
# args.add_argument('--pred_data', type=str, default='data/validation.json')
# args.add_argument('--eval_data', type=str, default='data/validation.json')
# args = args.parse_args()

evaluator = RougeEvaluator()

pred_files = ['data/test_output.json']

names = ['transformer']
eval_file = 'data/test.json'


model_scores = [[] for _ in range(len(names))]
for i, pred_file in enumerate(pred_files):
    with open(eval_file, 'r') as f:
        eval_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    assert len(eval_data) == len(pred_data)

    pred_sums = []
    eval_sums = []
    for eval, pred in tqdm.tqdm(zip(eval_data, pred_data), total=len(eval_data)):
        pred_sums.append(pred['summary'])
        eval_sums.append(eval['summary'])

    scores = evaluator.batch_score(pred_sums, eval_sums)

    for k, v in scores.items():
        val = "{:.3f} & {:.3f} & {:.3f}".format(v["p"], v["r"], v["f"])
        model_scores[i].append(val)

metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-4', 'ROUGE-l']
table_name = '\t' + '\t\t'.join(names)
print(table_name)
for k, metric in enumerate(metrics):
    print('{}: '.format(metric), end = '')
    for i, model in enumerate(names):
        print(model_scores[i][k], end = '')
    print()

