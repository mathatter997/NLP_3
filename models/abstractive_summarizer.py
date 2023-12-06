import tqdm
import random
import torch
import numpy as np
import os
import json


from models.seq2seq import Seq2SeqTransformer
from typing import Iterable, List
from collections import Counter


from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader




class AbstractiveSummarizer:

    def __init__(self, file, device):
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        self.device = device

        self.preprocess(file, train=True)
        self.SRC_VOCAB_SIZE = len(self.art_vocab)
        self.TGT_VOCAB_SIZE = len(self.sum_vocab)
        self.EMB_SIZE = 512
        self.NHEAD = 4
        self.FFN_HID_DIM = 128
        self.NUM_ENCODER_LAYERS = 3
        self.NUM_DECODER_LAYERS = 3

        self.transformer = Seq2SeqTransformer(self.NUM_ENCODER_LAYERS, self.NUM_DECODER_LAYERS, self.EMB_SIZE,
                                        self.NHEAD, self.SRC_VOCAB_SIZE, self.TGT_VOCAB_SIZE, self.FFN_HID_DIM)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)

        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    def preprocess(self, path, batch_size=16, train=False, n=-1):
        def build_vocab(filepath, tokenizer, specials=['<unk>', '<pad>', '<bos>', '<eos>']):
            article_counter = Counter()
            summary_counter = Counter()
            with open(filepath, 'r') as f:
                train_data = json.load(f)
            for article in train_data[:]:
                article_counter.update(tokenizer((article['article'])))
                summary_counter.update(tokenizer((article['summary'])))
            return vocab(article_counter, specials=specials, min_freq=30), vocab(summary_counter, specials=specials, min_freq=30)

        art_vocab, sum_vocab = build_vocab(path, self.en_tokenizer)
        art_vocab.set_default_index(art_vocab['<unk>'])
        sum_vocab.set_default_index(sum_vocab['<unk>'])

        if train:
            self.PAD_IDX = art_vocab['<pad>'] #PAD_IDX is the same for both vocabularies
            self.BOS_IDX = art_vocab['<bos>'] #BOS_IDX is the same for both vocabularies
            self.EOS_IDX = art_vocab['<eos>'] #EOS_IDX is the same for both vocabularies
            self.art_vocab = art_vocab
            self.sum_vocab = sum_vocab

        def data_process(filepath):
            with open(filepath, 'r') as f:
                train_data = json.load(f)

            data = []
            for article in train_data:
                article_tensor =  torch.tensor([self.art_vocab[token] for token in self.en_tokenizer(article['article'])],
                                    dtype=torch.long)
                summary_tensor = torch.tensor([self.sum_vocab[token] for token in self.en_tokenizer(article['summary'])],
                                    dtype=torch.long)
                data.append((article_tensor, summary_tensor))
            return data


        def generate_batch(data_batch):
            ar_batch, sum_batch = [], []
            for (ar_item, sum_item) in data_batch:
                ar_batch.append(torch.cat([torch.tensor([self.BOS_IDX]), ar_item, torch.tensor([self.EOS_IDX])], dim=0))
                sum_batch.append(torch.cat([torch.tensor([self.BOS_IDX]), sum_item, torch.tensor([self.EOS_IDX])], dim=0))
            ar_batch = pad_sequence(ar_batch, padding_value=self.PAD_IDX)
            sum_batch = pad_sequence(sum_batch, padding_value=self.PAD_IDX)
            return ar_batch, sum_batch

        train_data = data_process(path)

        if n < 0:
            n = len(train_data)
        train_data = train_data[:n]
        return DataLoader(train_data, batch_size=batch_size,
                                shuffle=True, collate_fn=generate_batch)

    def train(self, data, epochs=1):
        def train_epoch(model, optimizer):
            model.train()
            losses = 0

            for src, tgt in data:
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                tgt_input = tgt[:-1, :]


                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)
                logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

                optimizer.zero_grad()

                tgt_out = tgt[1:, :]
                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                loss.backward()

                optimizer.step()
                losses += loss.item()

            return losses / len(list(data))

        from timeit import default_timer as timer

        for epoch in range(1, epochs+1):
            start_time = timer()
            train_loss = train_epoch(self.transformer, self.optimizer)
            end_time = timer()
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    def predict(self, X):
        def sequential_transforms(*transforms):
            def func(txt_input):
                for transform in transforms:
                    txt_input = transform(txt_input)
                return txt_input
            return func

        # function to add BOS/EOS and create tensor for input sequence indices
        def tensor_transform(token_ids: List[int]):
            return torch.cat((torch.tensor([self.BOS_IDX]),
                            torch.tensor(token_ids),
                            torch.tensor([self.EOS_IDX])))

        # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
        text_transform_sum = sequential_transforms(self.en_tokenizer, #Tokenization
                                                self.sum_vocab, #Numericalization
                                                tensor_transform) # Add BOS/EOS and create tensor
        def greedy_decode(model, src, src_mask, max_len, start_symbol):
            src = src.to(self.device)
            src_mask = src_mask.to(self.device)

            memory = model.encode(src, src_mask)
            ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
            for i in range(max_len-1):
                memory = memory.to(self.device)
                tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                            .type(torch.bool)).to(self.device)
                out = model.decode(ys, memory, tgt_mask)
                out = out.transpose(0, 1)
                prob = model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()

                ys = torch.cat([ys,
                                torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
                if next_word == self.EOS_IDX:
                    break
            return ys

        # actual function to translate input sentence into target language
        def translate(model: torch.nn.Module, src_sentence: str):
            model.eval()
            src = text_transform_sum(src_sentence).view(-1, 1)
            num_tokens = src.shape[0]
            src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
            tgt_tokens = greedy_decode(
                model,  src, src_mask, max_len=num_tokens + 5, start_symbol=self.BOS_IDX).flatten()
            return " ".join(self.sum_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace("<unk>", "")

        summaries = []
        for article in X:
            summaries.append(translate(self.transformer, article))
        
        return summaries
    
    def evaluate(self, dataloader, mod=10):
        self.transformer.eval()
        losses = 0


        for i, (src, tgt) in enumerate(dataloader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)
            logits = self.transformer(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
            if i % mod:
                print(i, '/', len(dataloader), ':', losses)
        
        print(losses, len(list(dataloader)))
        return losses / len(list(dataloader))



    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = (src == self.PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

