from bert4keras.snippets import to_array
from bert4keras.snippets import sequence_padding
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
import re
import os
import numpy as np

class bertDeal(object):
    def __init__(self, bertpath='./datas/bert/chinese_L-12_H-768_A-12/'):
        self.bertpath = bertpath
        self.vector_len = int(re.split(r'_[A-Z]-', bertpath)[2])
        self.bert_vocab = os.path.join(bertpath, 'vocab.txt')
        self.bert_checkpoint = os.path.join(bertpath, 'bert_model.ckpt')
        self.bert_config = os.path.join(bertpath, 'bert_config.json')
        self.bertTokenizers = Tokenizer(self.bert_vocab, do_lower_case=True)  # 建立分词器
        self.bertModel = build_transformer_model(self.bert_config, self.bert_checkpoint)  # 建立模型，加载权重

    def toknizer(self, text=[]):
        token_ids = []
        segment_ids = []
        for i, te in enumerate(text):
            if i % 100 == 0:
                print(i, '/', len(text))
            t, s = self.bertTokenizers.encode(te)
            token_ids.append(t)
            segment_ids.append(s)
        pad_token_ids = sequence_padding(token_ids)
        pad_segment_ids = sequence_padding(segment_ids)
        return pad_token_ids, pad_segment_ids

    def embeding(self, text=[]):
        tokens, segments = self.toknizer(text)
        vectors = np.zeros(shape=(tokens.shape[0], tokens.shape[1], self.vector_len))
        for i in range(tokens.shape[0]):
            vectors[i] = np.reshape(self.bertModel.predict([tokens[i], segments[i]]),
                                    (tokens.shape[1], self.vector_len))
            if i % 10 == 0:
                print(i, vectors[i].shape)
        return vectors