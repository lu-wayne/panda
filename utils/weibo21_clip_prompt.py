
import pickle
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import torch
import pandas as pd
from torchvision import datasets, models, transforms
import os
import numpy as np
from PIL import Image
from openprompt.data_utils import InputExample
from openprompt.plms.mlm import MLMTokenizerWrapper
from openprompt.prompts import ManualTemplate
from transformers import RobertaTokenizer
def tmpt_text_prompting_encode(textual_tokenizer, sentences, targets):
    template_text = '{"placeholder":"text_a", "shortenable": True}. {"placeholder":"text_b", "shortenable": False} {"mask"}.'
    prompting_template = ManualTemplate(tokenizer=textual_tokenizer, text=template_text)
    #print(textual_tokenizer.model_max_length)
    #wrapped_tokenizer = MLMTokenizerWrapper(max_seq_length=textual_tokenizer.model_max_length, tokenizer=textual_tokenizer, truncate_method="tail")
    wrapped_tokenizer = MLMTokenizerWrapper(max_seq_length=197, tokenizer=textual_tokenizer, truncate_method="tail")
    input_ids = []
    loss_ids = []
    attention_mask = []
    for sentence, target in zip(sentences, targets):
        prompting_data = InputExample(text_a=sentence, text_b=target)
        encoding = wrapped_tokenizer.tokenize_one_example(prompting_template.wrap_one_example(prompting_data), teacher_forcing=False)
        #print("Input IDs:", encoding['input_ids'])
        input_ids.append(encoding['input_ids'])
        loss_ids.append(encoding['loss_ids'])
        attention_mask.append(encoding['attention_mask'])
    return {'input_ids': torch.tensor(input_ids), 'text_loss_ids': torch.tensor(loss_ids), 'attention_mask': torch.tensor(attention_mask)}
def read_image():
    image_list = {}
    file_list = ['data/nonrumor_images/', 'data/rumor_images/']
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        for i, filename in enumerate(os.listdir(path)):  # assuming gif

            # print(filename)
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im)
                #im = 1
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except:
                print("wrong"+filename)
    print("image length " + str(len(image_list)))
    #print("image names are " + str(image_list.keys()))
    return image_list



def read_image_prompt():
    image_list = {}
    file_list = ['/home/comp/yutong/MMDFND/imagePrompt/weibo21/']
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        for i, filename in enumerate(os.listdir(path)):  # assuming gif

            # print(filename)
            im = Image.open(path + filename).convert('RGB')
            im = data_transforms(im)
            #im = 1
            image_list[filename.split('/')[-1].split(".")[0].lower()] = im
    #print("image length " + str(len(image_list)))
    #print("image names are " + str(image_list.keys()))
    return image_list

def read_clip_image_prompt():
    image_list = {}
    file_list = ['/home/comp/yutong/MMDFND/imagePrompt/weibo21/']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    for path in file_list:
        for i, filename in enumerate(os.listdir(path)):  # assuming gif
                im = Image.open(path + filename)
                im = preprocess(im).unsqueeze(0).to(device)
                #im = 1
                image_list[filename.split('/')[-1].split(".")[0]] = im
    #print("image length " + str(len(image_list)))
    #print("image names are " + str(image_list.keys()))
    return image_list
def _init_fn(worker_id):
    np.random.seed(2024)

def read_pkl(path):
    with open(path,"rb")as f:
        t = pickle.load(f)
    return t
def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data

"""
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
"""
def word2input(texts,vocab_file,max_len,category):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    prompt = category
    for i in range(len(category)):
        prompt.iloc[i] = f"This {category.iloc[i]} domain news is "
    input_data = tmpt_text_prompting_encode(tokenizer, texts, prompt)
    token_ids =[]
    for i,text in enumerate(texts):
        token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i,token in enumerate(token_ids):
        masks[i] = (token != 0)
    return input_data["input_ids"],input_data["attention_mask"],input_data["text_loss_ids"],token_ids,masks

class bert_data():
    def __init__(self,max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict
        self.category_dict2 = {
        "科技": "0",
        "军事": "1",
        "教育考试": "2",
        "灾难事故": "3",
        "政治": "4",
        "医药健康": "5",
        "财经商业": "6",
        "文体娱乐": "7",
        "社会生活": "8"
        }
    def load_data(self,path,imagepath,clipimagepath,shuffle,text_only = False):
        self.data = pd.read_excel(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        content = self.data['content'].astype('object').to_numpy()
        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        category = torch.tensor(self.data['category'].astype('object').apply(lambda c: self.category_dict[c]).to_numpy())
        content_clip_domain = self.data['content'].astype('object')
        for i in range(len(self.data['category'])):
            content_clip_domain.iloc[i] = "This "+self.data['category'].iloc[i]+" domain news is "+content_clip_domain.iloc[i]
        content_clip_domain = content_clip_domain.astype('object').to_numpy()
        original_cata = self.data['category'].astype('object')
        input_ids,attention_mask,text_loss_ids, token_ids,masks = word2input(content,self.vocab_file,self.max_len,original_cata)
        #token_ids, masks = word2input(content,self.vocab_file,self.max_len)
        ordered_image = pickle.load(open(imagepath,'rb'))
        clip_image = pickle.load(open(clipimagepath, 'rb'))
        clip_text = clip.tokenize(content_clip_domain)
        datasets =TensorDataset(token_ids,
                                masks,
                                label,
                                category,
                                ordered_image,
                                clip_image,
                                clip_text,
                                input_ids,
                                attention_mask,
                                text_loss_ids

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
