
import os
from utils.clip_dataloader_prompt import bert_data as weibo_data
from utils.weibo21_clip_prompt import bert_data as weibo21_data
from utils.FINE_clip_prompt import bert_data as FINE_data
from model.PANDA import Trainer as LPROMPTIMAGETrainer
from model.FTmodel2 import Trainer as LPROMPTIMAGETrainer2
from model.clean_vib import Trainer as LPROMPTIMAGETrainer3


class Run():
    def __init__(self,
                 config
                 ):
        self.configinfo = config

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_type = config['emb_type']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']
        self.dataset = config['dataset']
        if config['dataset']=="weibo":
            self.dataset_type = "CN"
            self.root_path = './Weibo/'
            self.train_path = self.root_path + 'train_origin.csv'  # 如果9个领域就要改成train.csv
            self.val_path = self.root_path + 'val_origin.csv'  # 如果9个领域就要改成val.csv
            self.test_path = self.root_path + 'test_origin.csv'  # 如果9个领域就要改成test.csv
            self.category_dict = {
                "经济": 0,
                "健康": 1,
                "军事": 2,
                "科学": 3,
                "政治": 4,
                "国际": 5,
                "教育": 6,
                "娱乐": 7,
                "社会": 8
            }
        if config['dataset']=="weibo21":
            self.dataset_type = "CN"
            self.root_path = './Weibo_21/'
            self.train_path = self.root_path + 'train_datasets.xlsx'#weibo21
            self.val_path = self.root_path + 'val_datasets.xlsx'#weibo21
            self.test_path = self.root_path + 'test_datasets.xlsx'#weibo21
            self.category_dict = {
                "科技": 0,
                "军事": 1,
                "教育考试": 2,
                "灾难事故": 3,
                "政治": 4,
                "医药健康": 5,
                "财经商业": 6,
                "文体娱乐": 7,
                "社会生活": 8
            }
        if config['dataset'] == "pheme":
            self.root_path = './pheme/'
            self.train_path = self.root_path + 'train.csv'#weibo21
            self.val_path = self.root_path + 'val.csv'#weibo21
            self.test_path = self.root_path + 'test.csv'#weibo21
            self.category_dict = {
                "society": 0,
                "military": 1,
                "international": 2,
                "entertainment": 3,
                "politics": 4
            }
        if config['dataset']=="FINE":
            self.dataset_type = "EN"
            self.root_path = './FINE/'
            self.train_path = self.root_path + 'FineFake_train.pkl'#weibo21
            self.val_path = self.root_path + 'FineFake_val.pkl'#weibo21
            self.test_path = self.root_path + 'FineFake_test.pkl'#weibo21
            self.category_dict = {
                "Society": 0,
                "Conflict": 1,
                "Politics": 2,
                "Entertainment": 3,
                "Health": 4,
                "Business": 5,
                "Uncategorized": 6
            }
    def get_dataloader(self,dataset):
        if self.emb_type == 'bert':
            if dataset =="weibo":
                loader = weibo_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                              category_dict=self.category_dict, num_workers=self.num_workers)
            if dataset =="weibo21":
                loader = weibo21_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                              category_dict=self.category_dict, num_workers=self.num_workers)
            if dataset =="pheme":
                loader = pheme_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                              category_dict=self.category_dict, num_workers=self.num_workers)
            if dataset =="FINE":
                loader = FINE_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                              category_dict=self.category_dict, num_workers=self.num_workers)
        #clip_weibo
        if dataset =="weibo":
            train_loader = loader.load_data(self.train_path,'Weibo/train_loader.pkl','Weibo/train_clip_loader.pkl',True)
            val_loader = loader.load_data(self.val_path,'Weibo/val_loader.pkl','Weibo/val_clip_loader.pkl',False)
            test_loader = loader.load_data(self.test_path,'Weibo/test_loader.pkl','Weibo/test_clip_loader.pkl',False)
        # clip_weibo21
        if dataset =="weibo21":
            val_loader = loader.load_data(self.val_path, 'Weibo_21/val_loader.pkl', 'Weibo_21/val_clip_loader.pkl', False)
            test_loader = loader.load_data(self.test_path, 'Weibo_21/test_loader.pkl', 'Weibo_21/test_clip_loader.pkl', False)
            train_loader = loader.load_data(self.train_path, 'Weibo_21/train_loader.pkl', 'Weibo_21/train_clip_loader.pkl', True)
        if dataset =="pheme":
            train_loader = loader.load_data(self.train_path, 'pheme/train_loader.pkl', 'pheme/train_clip_loader.pkl', True)
            val_loader = loader.load_data(self.val_path, 'pheme/val_loader.pkl', 'pheme/val_clip_loader.pkl', True)
            test_loader = loader.load_data(self.test_path, 'pheme/test_loader.pkl', 'pheme/test_clip_loader.pkl', True)
        if dataset =="FINE":
            train_loader = loader.load_data(self.train_path, 'FINE/train_loader.pkl', 'FINE/train_clip_loader.pkl', True)
            val_loader = loader.load_data(self.val_path, 'FINE/val_loader.pkl', 'FINE/val_clip_loader.pkl', True)
            test_loader = loader.load_data(self.test_path, 'FINE/test_loader.pkl', 'FINE/test_clip_loader.pkl', True)
        return train_loader, val_loader, test_loader

    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        train_loader, val_loader, test_loader = self.get_dataloader(self.dataset)
        if self.model_name == 'FTmodel':
            trainer = LPROMPTIMAGETrainer(emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, bert=self.bert,
                                    use_cuda=self.use_cuda, lr=self.lr, train_loader=train_loader, dropout=self.dropout,
                                    weight_decay=self.weight_decay, val_loader=val_loader, test_loader=test_loader,
                                    category_dict=self.category_dict, early_stop=self.early_stop, epoches=self.epoch,
                                    save_param_dir=os.path.join(self.save_param_dir, self.model_name),dataset_type=self.dataset_type)
        if self.model_name == 'FTmodel2':
            trainer = LPROMPTIMAGETrainer2(emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, bert=self.bert,
                                    use_cuda=self.use_cuda, lr=self.lr, train_loader=train_loader, dropout=self.dropout,
                                    weight_decay=self.weight_decay, val_loader=val_loader, test_loader=test_loader,
                                    category_dict=self.category_dict, early_stop=self.early_stop, epoches=self.epoch,
                                    save_param_dir=os.path.join(self.save_param_dir, self.model_name),dataset_type=self.dataset_type)
        if self.model_name == 'clean_vib':
            trainer = LPROMPTIMAGETrainer3(emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, bert=self.bert,
                                    use_cuda=self.use_cuda, lr=self.lr, train_loader=train_loader, dropout=self.dropout,
                                    weight_decay=self.weight_decay, val_loader=val_loader, test_loader=test_loader,
                                    category_dict=self.category_dict, early_stop=self.early_stop, epoches=self.epoch,
                                    save_param_dir=os.path.join(self.save_param_dir, self.model_name),dataset_type=self.dataset_type)
        trainer.train()
