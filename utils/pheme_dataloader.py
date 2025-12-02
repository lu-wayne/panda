
import pickle
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer,RobertaTokenizer
import torch
import pandas as pd
from torchvision import datasets, models, transforms
import os
import numpy as np
from PIL import Image


def _init_fn(worker_id):
    np.random.seed(2024)

def read_pkl(path):
    with open(path,"rb")as f:
        t = pickle.load(f)
    return t
def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data

def word2input(texts,vocab_file,max_len):
    #tokenizer = BertTokenizer(vocab_file=vocab_file)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    token_ids =[]
    for i,text in enumerate(texts):
        token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i,token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids,masks

class bert_data():
    def __init__(self,max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data(self,path,imagepath,clipimagepath,shuffle,text_only = False):
        self.data = pd.read_csv(path,encoding='utf-8')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        content = self.data['text'].astype('object').to_numpy()
        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        category = torch.tensor(self.data['domain'].astype('object').apply(lambda c: self.category_dict[c]).to_numpy())
        token_ids, masks = word2input(content,self.vocab_file,self.max_len)
        ordered_image = pickle.load(open(imagepath,'rb'))
        clip_image = pickle.load(open(clipimagepath, 'rb'))
        clip_text = clip.tokenize(content)
        datasets =TensorDataset(token_ids,
                                masks,
                                label,
                                category,
                                ordered_image,
                                clip_image,
                                clip_text
        )
        dataloader = DataLoader(
            dataset = datasets,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            shuffle = shuffle,
            worker_init_fn = _init_fn
        )
        return dataloader
