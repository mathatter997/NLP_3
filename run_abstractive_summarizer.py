import argparse
import json
from models.abstractive_summarizer import AbstractiveSummarizer
import torch

args = argparse.ArgumentParser()
args.add_argument('--train_data', type=str, default='data/train.json')
args.add_argument('--validation_data', type=str, default='data/validation.json')
args.add_argument('--eval_data', type=str, default='data/test.json')
args = args.parse_args()


with open(args.train_data, 'r') as f:
    train_data = json.load(f)

with open(args.validation_data, 'r') as f:
    validation_data = json.load(f)

# train_articles = [article['article'] for article in train_data]
# train_summaries = [article['summary'] for article in train_data]

# val_articles = [article['article'] for article in validation_data]
# val_summaries = [article['summary'] for article in validation_data]





DEVICE = torch.device('cpu')
model = AbstractiveSummarizer('data/test.json', DEVICE)

print(model.SRC_VOCAB_SIZE, model.TGT_VOCAB_SIZE)
model.transformer.load_state_dict(torch.load('model', map_location='cpu'))

# model.train(train_articles, train_summaries, val_articles, val_summaries)

with open(args.eval_data, 'r') as f:
    eval_data = json.load(f)


# train_data = model.preprocess('data/train.json', n=10)
# summaries  = model.predict(train_articles[:10])

eval_summaries = [article['summary'] for article in eval_data]
eval_articles = [article['article'] for article in eval_data]

import time

start = time.time()
summaries = model.predict(eval_articles)
end = time.time()

print(end - start)

# eval_out_data = [{'train_summary': train_summary, 'summary': summary} for train_summary, summary in zip(train_summaries, summaries)]


# with open('data/train_output.json', 'w') as f:
#     json.dump(eval_out_data, f)