
import pickle
from LongCLIP.model import longclip as clip
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from transformers import RobertaTokenizer
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

def word2inputROBERT(texts, max_len):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def word2input(texts,vocab_file,max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
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
        with open(path, 'rb') as file:
            data = pickle.load(file)
        self.data = pd.DataFrame(data)
        if imagepath == "FINE/val_loader.pkl":
            self.data['text'].iloc[2410] = "Two business associates of President Trump’s personal lawyer Rudolph W. Giuliani have been charged with a scheme to route foreign money into U.S. elections. The two men, who helped Giuliani investigate former vice president Joe Biden, were arrested Wednesday night in Virginia, according to a person familiar with the charges. Lev Parnas and Igor Fruman have been under investigation by the U.S. attorney’s office in Manhattan and are expected to appear in federal court in Virginia later Thursday."
        self.data['text'] = self.data['text'].apply(lambda x: x[:500] if len(x) > 500 else x)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        content = self.data['text'].astype('object').to_numpy()
        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        category = torch.tensor(self.data['topic'].astype('object').apply(lambda c: self.category_dict[c]).to_numpy())
        #token_ids, masks = word2input(content,self.vocab_file,self.max_len)
        token_ids, masks = word2inputROBERT(content, self.max_len)
        ordered_image = pickle.load(open(imagepath,'rb'))
        clip_image = pickle.load(open(clipimagepath, 'rb'))
        clip_text = clip.tokenize(content)
        #("token_ids",token_ids.size())
        #print("masks", masks.size())
        #print("label", label.size())
        #print("category", category.size())
        #print("ordered_image", ordered_image.size())
        #print("clip_image", clip_image.size())
        #print("clip_text", clip_text.size())
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
