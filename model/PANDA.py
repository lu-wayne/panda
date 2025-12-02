import os
import tqdm
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
# from positional_encodings.torch_encodings import PositionalEncoding1D
import models_mae
from utils.utils import data2gpu, Averager, metrics, Recorder, clipdata2gpu
from utils.utils import metricsTrueFalse
from .layers import *
from .pivot import *
from timm.models.vision_transformer import Block
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import math


class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        return torch.sum(x,(1))/(x.shape[1])

    def sigma(self, x):
        return torch.sqrt((torch.sum((x.permute([1,0])-self.mu(x)).permute([1,0])**2,(1))+0.000000023)/(x.shape[1]))

    def forward(self, x, mu, sigma):
        # print(mu.shape) # 12
        x_mean = self.mu(x)
        x_std = self.sigma(x)
        x_reduce_mean = x.permute([1, 0]) - x_mean
        x_norm = x_reduce_mean/x_std
        # print(x_mean.shape) # 768, 12
        return (sigma.squeeze(1)*(x_norm + mu.squeeze(1))).permute([1,0])


class DomainCollaborativeAttention(nn.Module):
    """
    Domain-Collaborative Attention (DCA) module.
    Fuses knowledge from selected neighbor domains using cross-attention.
    The target domain's prompts act as Query, and prompts from the neighbor
    domains serve as Key and Value.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Args:
            query (Tensor): Target domain prompts (Batch, L_q, D_p)
            key (Tensor): Neighbor domain prompts (Batch, L_k, D_p)
            value (Tensor): Neighbor domain prompts (Batch, L_k, D_p)
            key_padding_mask (Tensor): Mask for non-selected domains (Batch, L_k)
        Returns:
            Tensor: Collaborated knowledge vector (Batch, L_q, D_p)
        """
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # Reshape for multi-head attention
        batch_size = Q.size(0)
        Q = Q.view(batch_size, -1, self.nhead, self.d_model // self.nhead).transpose(1, 2)
        K = K.view(batch_size, -1, self.nhead, self.d_model // self.nhead).transpose(1, 2)
        V = V.view(batch_size, -1, self.nhead, self.d_model // self.nhead).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

        if key_padding_mask is not None:
            # Expand mask for multi-head attention
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # (Batch, 1, 1, L_k)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_proj(context)



class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout, domain_num=9):
        super(MultiDomainPLEFENDModel, self).__init__()
        self.num_expert = 6
        self.task_num = 2
        # self.domain_num = 9
        # Per user request, the original domain_num is confusing, so we use a clear parameter
        self.domain_num = domain_num
        self.gate_num = 3
        self.num_share = 1
        self.unified_dim, self.text_dim = emb_dim, 768
        self.image_dim = 768
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.text_token_len = 197
        self.image_token_len = 197

        # ==================== Existing Expert Networks (No Changes) ====================
        text_expert_list = []
        for i in range(self.task_num): # Use task_num for compatibility with original code structure
            text_expert = []
            for j in range(self.num_expert):
                text_expert.append(cnn_extractor(emb_dim, feature_kernel))
            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
        self.text_experts = nn.ModuleList(text_expert_list)

        image_expert_list = []
        for i in range(self.task_num):
            image_expert = []
            for j in range(self.num_expert):
                image_expert.append(cnn_extractor(self.image_dim, feature_kernel))
            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)
        self.image_experts = nn.ModuleList(image_expert_list)

        fusion_expert_list = []
        for i in range(self.task_num):
            fusion_expert = []
            for j in range(self.num_expert):
                expert = nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320))
                fusion_expert.append(expert)
            fusion_expert = nn.ModuleList(fusion_expert)
            fusion_expert_list.append(fusion_expert)
        self.fusion_experts = nn.ModuleList(fusion_expert_list)

        final_expert_list = []
        for i in range(self.task_num):
            final_expert = []
            for j in range(self.num_expert):
                final_expert.append(Block(dim=320, num_heads=8))
            final_expert = nn.ModuleList(final_expert)
            final_expert_list.append(final_expert)
        self.final_experts = nn.ModuleList(final_expert_list)

        text_share_expert, image_share_expert, fusion_share_expert,final_share_expert = [], [], [],[]
        for i in range(self.num_share):
            text_share, image_share, fusion_share, final_share = [], [], [], []
            for j in range(self.num_expert*2):
                text_share.append(cnn_extractor(emb_dim, feature_kernel))
                image_share.append(cnn_extractor(self.image_dim, feature_kernel))
                expert = nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320))
                fusion_share.append(expert)
                final_share.append(Block(dim=320, num_heads=8))
            self.text_share_expert.append(nn.ModuleList(text_share))
            self.image_share_expert.append(nn.ModuleList(image_share))
            self.fusion_share_expert.append(nn.ModuleList(fusion_share))
            self.final_share_expert.append(nn.ModuleList(final_share))
        self.text_share_expert = nn.ModuleList(text_share_expert)
        self.image_share_expert = nn.ModuleList(image_share_expert)
        self.fusion_share_expert = nn.ModuleList(fusion_share_expert)
        self.final_share_expert = nn.ModuleList(final_share_expert)

        image_gate_list, text_gate_list, fusion_gate_list, fusion_gate_list0,final_gate_list = [], [], [], [],[]
        for i in range(self.task_num):
            image_gate_list.append(nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(), nn.Linear(self.unified_dim, self.num_expert * 3), nn.Dropout(0.1), nn.Softmax(dim=1)))
            text_gate_list.append(nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(), nn.Linear(self.unified_dim, self.num_expert * 3), nn.Dropout(0.1), nn.Softmax(dim=1)))
            fusion_gate_list.append(nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(), nn.Linear(self.unified_dim, self.num_expert * 4), nn.Dropout(0.1), nn.Softmax(dim=1)))
            fusion_gate_list0.append(nn.Sequential(nn.Linear(320, 160), nn.SiLU(), nn.Linear(160, self.num_expert * 3), nn.Dropout(0.1), nn.Softmax(dim=1)))
            final_gate_list.append(nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 160), nn.SiLU(), nn.Linear(160, self.num_expert * 3), nn.Dropout(0.1), nn.Softmax(dim=1)))
        self.image_gate_list = nn.ModuleList(image_gate_list)
        self.text_gate_list = nn.ModuleList(text_gate_list)
        self.fusion_gate_list = nn.ModuleList(fusion_gate_list)
        self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0)
        self.final_gate_list = nn.ModuleList(final_gate_list)

        self.text_attention = MaskAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.fusion_attention = TokenAttention(self.unified_dim * 2)
        self.final_attention = TokenAttention(320)

        # ==================== Existing Classifiers for Auxiliary Tasks (No Changes) ====================
        self.text_classifier = MLP(320, mlp_dims, dropout)
        self.image_classifier = MLP(320, mlp_dims, dropout)
        self.fusion_classifier = MLP(320, mlp_dims, dropout)
        
        # self.max_classifier will be replaced by PANDA's final classification head
        # self.max_classifier = MLP(320 * 1, mlp_dims, dropout) 

        self.MLP_fusion = MLP_fusion(960, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(320, 320, [348], 0.1)
        self.MLP_fusion0 = MLP_fusion(768 * 2, 768, [348], 0.1)
        self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)
        self.att_mlp_text = MLP_fusion(320, 2, [174], 0.1)
        self.att_mlp_img = MLP_fusion(320, 2, [174], 0.1)
        self.att_mlp_mm = MLP_fusion(320, 2, [174], 0.1)
        
        self.model_size = "base"
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        self.image_model.cuda()
        checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)
        for param in self.image_model.parameters():
            param.requires_grad = False
        
        # ==================== Existing Graph/Pivot/Transformer parts  ====================
        self.ClipModel,_ = load_from_name("ViT-B-16", device="cuda", download_root='./')
        feature_emb_size, img_emb_size, text_emb_size = 320, 320, 320
        feature_num = 4
        self.layers = 12
        self.transformers = torch.nn.ModuleList([TransformerLayer(feature_emb_size, head_num=4, dropout=0.6) for _ in range(self.layers)])

        self.mlp_img = torch.nn.ModuleList([MLP_trans(img_emb_size, feature_emb_size, dropout=0.6) for _ in range(feature_num)])
        self.mlp_text = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in range(feature_num)])
        

        # PANDA Hyperparameters 
        self.p_Lp = 4  # Number of prompts per modality view
        self.p_Dp = 320 # Dimension of prompt vectors, matching feature dim
        self.p_M = 16  # Number of prototypes per domain
        self.p_S = 2   # Number of top-S neighbor domains to select
        
        # 2.2 Domain-aware Modal Prompt Generation (DMPG)
        # Prompts for text, vision, and multimodal views are concatenated
        self.domain_modal_prompts = nn.Parameter(torch.randn(self.domain_num, 3 * self.p_Lp, self.p_Dp))

        # 2.3 Prototype-based Asymmetric Distance (PAD)
        # 2.3.1 Domain Prototype Learning
        self.domain_prototypes = nn.Parameter(torch.randn(self.domain_num, self.p_M, 320))
        # Autoencoder for prototype learning
        self.proto_encoder = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(320, 320))
        self.proto_decoder = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(320, 320))

        # 2.4 Dynamic Neighbor-Domain Selection and Fusion
        # 2.4.2 Domain-Collaborative Attention (DCA)
        self.dca_module = DomainCollaborativeAttention(d_model=self.p_Dp, nhead=8, dropout=0.1)

        # 2.5 Prediction and Loss Function
        # 2.5.1 Final Classification Head
        self.final_classifier_panda = MLP(320 + self.p_Dp, mlp_dims, dropout) # h_di (320) + h_collab (Dp)
        


    def _calculate_pad(self):
        """
        Helper function to calculate Prototype-based Asymmetric Distance (PAD) matrix.
        dist(s->t) measures how well source domain s's prototypes fit into target domain t.
        """
        protos = self.domain_prototypes # (D, M, Dim)
        # Expand dims for broadcasting: protos1 -> (D, 1, M, Dim), protos2 -> (1, D, M, Dim)
        protos1 = protos.unsqueeze(1)
        protos2 = protos.unsqueeze(0)
        
        # Calculate pairwise L2 distance between all prototypes of all domains
        # Shape: (D_s, D_t, M_s, M_t)
        pairwise_dist = torch.cdist(protos1, protos2, p=2.0)
        
        # For each source prototype, find the min distance to any target prototype
        # Shape: (D_s, D_t, M_s)
        min_dist_to_target, _ = torch.min(pairwise_dist, dim=3)
        
        # Average these minimum distances over all source prototypes
        # This yields the asymmetric distance dist(s->t)
        # Shape: (D_s, D_t)
        asymmetric_dist_matrix = torch.mean(min_dist_to_target, dim=2)
        
        return asymmetric_dist_matrix

    def _gumbel_neighbor_selector(self, sim_matrix, target_domain_indices):
        """
        Gumbel-based Neighbor Selector (GNS) to select top-S neighbors.
        Uses Gumbel-top-k trick for differentiable selection.
        """
        # Get similarity scores for each item in the batch to all source domains
        # sim_matrix is (D_s, D_t) -> we need sim(s->t) for each t in batch
        batch_sims = sim_matrix[:, target_domain_indices].transpose(0, 1) # (Batch, D_s)

        # Gumbel-top-k trick
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(batch_sims) + 1e-9) + 1e-9)
        perturbed_scores = torch.log(batch_sims + 1e-9) + gumbel_noise
        
        # Select top-S neighbors based on perturbed scores
        _, top_indices = torch.topk(perturbed_scores, self.p_S, dim=1) # (Batch, S)
        
        return top_indices

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        text_feature = self.bert(inputs, attention_mask=masks)[0]  # ([64, 197, 768])
        image = kwargs['image']
        image_feature = self.image_model.forward_ying(image)  # ([64, 197, 768])
        clip_image = kwargs['clip_image']
        clip_text = kwargs['clip_text']
        domain_indices = kwargs['category'] # Assuming 'category' is the domain index

        # ==================== Existing Feature Extraction (No Changes) ====================
        with torch.no_grad(): 
            clip_image_feature = self.ClipModel.encode_image(clip_image)
            clip_text_feature = self.ClipModel.encode_text(clip_text)
            clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)
            clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True)
        clip_fusion_feature = torch.cat((clip_image_feature, clip_text_feature), dim=-1)
        clip_fusion_feature = self.clip_fusion(clip_fusion_feature.float())

        text_atn_feature = self.text_attention(text_feature, masks)
        image_atn_feature, _ = self.image_attention(image_feature)
        fusion_feature = torch.cat((image_feature, text_feature), dim=-1)
        fusion_atn_feature, _ = self.fusion_attention(fusion_feature)
        fusion_atn_feature = self.MLP_fusion0(fusion_atn_feature)

        text_gate_input, image_gate_input, fusion_gate_input = text_atn_feature, image_atn_feature, fusion_atn_feature
        
        # Using domain 0 gates as per original logic for expert selection
        text_gate_out = self.text_gate_list[0](text_gate_input)
        image_gate_out = self.image_gate_list[0](image_gate_input)

        text_experts_feature = 0
        text_gate_share_expert_value = 0
        for j in range(self.num_expert):
            text_experts_feature += (self.text_experts[0][j](text_feature) * text_gate_out[:, j].unsqueeze(1))
        for j in range(self.num_expert * 2):
            tmp_expert = self.text_share_expert[0][j](text_feature)
            text_experts_feature += (tmp_expert * text_gate_out[:, (self.num_expert + j)].unsqueeze(1))
            text_gate_share_expert_value += (tmp_expert * text_gate_out[:, (self.num_expert + j)].unsqueeze(1))
        
        att_text = F.softmax(self.att_mlp_text(text_experts_feature), dim=-1)
        text_gate_expert_value = [att_text[:, 0].view(-1, 1) * text_experts_feature, att_text[:, 1].view(-1, 1) * text_experts_feature]

        image_experts_feature = 0
        image_gate_share_expert_value = 0
        for j in range(self.num_expert):
            image_experts_feature += (self.image_experts[0][j](image_feature) * image_gate_out[:, j].unsqueeze(1))
        for j in range(self.num_expert * 2):
            tmp_expert = self.image_share_expert[0][j](image_feature)
            image_experts_feature += (tmp_expert * image_gate_out[:, (self.num_expert + j)].unsqueeze(1))
            image_gate_share_expert_value += (tmp_expert * image_gate_out[:, (self.num_expert + j)].unsqueeze(1))

        att_img = F.softmax(self.att_mlp_img(image_experts_feature), dim=-1)
        image_gate_expert_value = [att_img[:, 0].view(-1, 1) * image_experts_feature, att_img[:, 1].view(-1, 1) * image_experts_feature]

        fusion_share_feature = self.MLP_fusion(torch.cat((clip_fusion_feature, text_gate_share_expert_value, image_gate_share_expert_value), dim=-1))
        fusion_gate_out0 = self.fusion_gate_list0[0](self.domain_fusion(fusion_share_feature))
        
        fusion_experts_feature = 0
        for n in range(self.num_expert):
            fusion_experts_feature += (self.fusion_experts[0][n](fusion_share_feature) * fusion_gate_out0[:, n].unsqueeze(1))
        for n in range(self.num_expert * 2):
            fusion_experts_feature += (self.fusion_share_expert[0][n](fusion_share_feature) * fusion_gate_out0[:, (self.num_expert + n)].unsqueeze(1))
        
        att_mm = F.softmax(self.att_mlp_mm(fusion_experts_feature), dim=-1)
        fusion_gate_expert_value0 = [att_mm[:, 0].view(-1, 1) * fusion_experts_feature, att_mm[:, 1].view(-1, 1) * fusion_experts_feature]

        text_features = text_gate_expert_value[0]
        image_features = image_gate_expert_value[0]
        fusion_features = fusion_gate_expert_value0[0]

        # Auxiliary task predictions (no changes)
        text_fake_news_logits = self.text_classifier(text_features).squeeze(1)
        image_fake_news_logits = self.image_classifier(image_features).squeeze(1)
        fusion_fake_news_logits = self.fusion_classifier(fusion_features).squeeze(1)
        text_fake_news = torch.sigmoid(text_fake_news_logits)
        image_fake_news = torch.sigmoid(image_fake_news_logits)
        fusion_fake_news = torch.sigmoid(fusion_fake_news_logits)
        
        # This is h_{d_i} 
        h_di = text_features + image_features + fusion_features



        # 2.3.1 Prototype Learning and Reconstruction Loss Calculation
        encoded_h = self.proto_encoder(h_di)
        # For each sample, find the closest prototype within its own domain
        protos_for_batch = self.domain_prototypes[domain_indices] # (Batch, M, Dim)
        dist_to_protos = torch.cdist(encoded_h.unsqueeze(1), protos_for_batch) # (Batch, 1, M)
        _, closest_proto_indices = torch.min(dist_to_protos, dim=2) # (Batch, 1)
        
        # Gather the quantized vectors
        quantized_h = protos_for_batch.gather(1, closest_proto_indices.unsqueeze(2).expand(-1, -1, h_di.shape[1])).squeeze(1)
        
        reconstructed_h = self.proto_decoder(quantized_h)
        loss_rec = F.mse_loss(reconstructed_h, h_di)

        # 2.3.2 Calculate Prototype-based Asymmetric Distance (PAD)
        dist_matrix = self._calculate_pad() # (D_s, D_t)

        # 2.4.1 Gumbel-based Neighbor Selector (GNS)
        sim_matrix = 1.0 / (dist_matrix + 1e-6)
        neighbor_indices = self._gumbel_neighbor_selector(sim_matrix, domain_indices) # (Batch, S)


        # 2.4.2 Domain-Collaborative Attention (DCA)
        # Prepare Query, Key, Value for DCA
        # Query: prompts of the target domains for the batch
        target_prompts = self.domain_modal_prompts[domain_indices] # (Batch, 3*Lp, Dp)
        
        # Key/Value: prompts from the selected neighbor domains
        batch_size = h_di.size(0)
        # Gather neighbor prompts for each item in the batch
        neighbor_prompts = self.domain_modal_prompts[neighbor_indices.view(-1)].view(batch_size, self.p_S, 3 * self.p_Lp, self.p_Dp)
        # Reshape to (Batch, S * 3*Lp, Dp) for attention
        neighbor_prompts_flat = neighbor_prompts.view(batch_size, -1, self.p_Dp)

        # Create a mask to attend only to neighbor prompts (all are valid here)
        key_padding_mask = torch.ones(batch_size, self.p_S * 3 * self.p_Lp, device=h_di.device)

        # Perform attention
        collaborated_prompts = self.dca_module(target_prompts, neighbor_prompts_flat, neighbor_prompts_flat, key_padding_mask)
        # Aggregate to get the final collaborated knowledge vector
        h_collab = torch.mean(collaborated_prompts, dim=1) # (Batch, Dp)

        # 2.5.1 Final Classification
        h_final = torch.cat([h_di, h_collab], dim=1)
        fake_news_logits = self.final_classifier_panda(h_final).squeeze(1)
        fake_news_sigmoid = torch.sigmoid(fake_news_logits)
        

        return fake_news_sigmoid, text_fake_news, image_fake_news, fusion_fake_news, loss_rec





class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 bert,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 loss_weight=[1, 0.006, 0.009, 5e-5],
                 lambda_rec=0.1, # Hyperparameter for reconstruction loss
                 domain_num=9, # Number of domains
                 early_stop=5,
                 epoches=100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda
        self.lambda_rec = lambda_rec
        self.domain_num = domain_num

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        if not os.path.exists(save_param_dir):
            self.save_param_dir = os.makedirs(save_param_dir)
        else:
            self.save_param_dir = save_param_dir

    def train(self):
        # Pass domain_num to the model
        self.model = MultiDomainPLEFENDModel(self.emb_dim, self.mlp_dims, self.bert, 320, self.dropout, self.domain_num)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = clipdata2gpu(batch)
                label = batch_data['label']

                # The model now returns an additional loss_rec term
                label0, text_fake_news, image_fake_news, fusion_fake_news, loss_rec = self.model(**batch_data)
                
                # Main classification loss (L_cls)
                loss0 = loss_fn(label0, label.float())

                # Auxiliary task losses (as before)
                loss12 = loss_fn(text_fake_news, label.float())
                loss22 = loss_fn(image_fake_news, label.float())
                loss32 = loss_fn(fusion_fake_news, label.float())
                
                # Total loss computation L_total = L_cls + λ * L_rec
                loss = loss0 + self.lambda_rec * loss_rec + (loss12 + loss22 + loss32) / 3.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            results0 = self.test(self.val_loader) # Validate on val_loader
            mark = recorder.add(results0)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter_panda.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_panda.pkl')))
        print("开始进行最后的测试: ")
        results0 = self.test(self.test_loader)
        print("最后的结果", results0)

        return results0, os.path.join(self.save_param_dir, 'parameter_panda.pkl')


    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = clipdata2gpu(batch)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                
                # Adjust for the new model output signature
                batch_label_pred, _, _, _, _ = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        metric_res = metricsTrueFalse(label, pred, category, self.category_dict)
        return metric_res
    
