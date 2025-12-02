from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import os
import numpy as np
import torch
import pickle
from PIL import Image
from LongCLIP.model import longclip
from transformers import BertTokenizer
import cn_clip.clip as clip
from torchvision import datasets, models, transforms
from cn_clip.clip import load_from_name, available_models
def read_image():
    image_list = {}
    file_list = ['FINE/Image/washingtonpost/','FINE/Image/twitter/','FINE/Image/snope/','FINE/Image/reddit/','FINE/Image/photoshopbattles/','FINE/Image/oldphotosinreallife/','FINE/Image/nytimes/','FINE/Image/misleadingthumbnails/','FINE/Image/historyanecdotes/','FINE/Image/HeresAFunFact/','FINE/Image/conspiracy/','FINE/Image/cnn/','FINE/Image/cdc_gov/','FINE/Image/apnews/']
    #file_list = ['FINE/Image/washingtonpost/','FINE/Image/cdc_gov/']
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        for i, filename in enumerate(os.listdir(path)):  # assuming gif
            im = Image.open(path + filename).convert('RGB')
            im = data_transforms(im)
            #im = 1
            image_list[path[5:]+filename.split('/')[-1].split(".")[0]] = im
    print("image length " + str(len(image_list)))
    #print("image names are " + str(image_list.keys()))
    return image_list

def _init_fn(worker_id):
    np.random.seed(2021)

def read_pkl(path):
    with open(path,"rb")as f:
        t = pickle.load(f)
    return t
def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data

class bert_data():
    def __init__(self,max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data_train(self,path,shuffle,text_only = False):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        df = pd.DataFrame(data)
        post = df
        #self.data = df_filter(read_pkl(path))
        ordered_image = []
        image_id_list = []
        image_id = ""
        image = read_image()
        for i, id in enumerate(post['text']):
            image_id = post.iloc[i]['image_path'].split(".")[0]
            if image_id not in image:
                print(image_id)
                print(i)
                print("not in!")
            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])

        #ordered_image = torch.tensor(list(ordered_image))
        ordered_image = torch.tensor([item.cpu().detach().numpy() for item in ordered_image])
        print(ordered_image.size())
        with open('FINE/train_loader.pkl', 'wb') as file:
            pickle.dump(ordered_image, file)
        return 1
    def load_data_test(self,path,shuffle,text_only = False):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        df = pd.DataFrame(data)
        post = df
        #self.data = df_filter(read_pkl(path))
        ordered_image = []
        image_id_list = []
        image_id = ""
        image = read_image()
        for i, id in enumerate(post['text']):
            image_id = post.iloc[i]['image_path'].split(".")[0]
            if image_id not in image:
                print(image_id)
                print(i)
                print("not in!")
            if image_id not in image:
                print("not in!")
            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])

        #ordered_image = torch.tensor(list(ordered_image))
        ordered_image = torch.tensor([item.cpu().detach().numpy() for item in ordered_image])
        print(ordered_image.size())
        with open('FINE/test_loader.pkl', 'wb') as file:
            pickle.dump(ordered_image, file)
        return 1
    def load_data_val(self,path,shuffle,text_only = False):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        df = pd.DataFrame(data)
        post = df
        #self.data = df_filter(read_pkl(path))
        ordered_image = []
        image_id_list = []
        image_id = ""
        image = read_image()
        for i, id in enumerate(post['text']):
            image_id = post.iloc[i]['image_path'].split(".")[0]
            if image_id not in image:
                print(image_id)
                print(i)
                print("not in!")
            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])

        #ordered_image = torch.tensor(list(ordered_image))
        ordered_image = torch.tensor([item.cpu().detach().numpy() for item in ordered_image])
        print(ordered_image.size())
        with open('FINE/val_loader.pkl', 'wb') as file:
            pickle.dump(ordered_image, file)
        return 1
category_dict = {
        "经济": 0,
        "健康": 1,
        "军事": 2,
        "科学": 3,
        "政治": 4,
        "教育": 5,
        "娱乐": 6,
        "社会": 7
}
loader = bert_data(max_len=170, batch_size=64, vocab_file='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt',
                   category_dict=category_dict, num_workers=1)
val_loader = loader.load_data_val("FINE/FineFake_val.pkl", True)#torch.Size([615, 3, 224, 224])
test_loader = loader.load_data_test("FINE/FineFake_test.pkl", True)#torch.Size([615, 3, 224, 224])
train_loader = loader.load_data_train("FINE/FineFake_train.pkl", True)#torch.Size([4926, 3, 224, 224])

