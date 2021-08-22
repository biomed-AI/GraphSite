import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Self_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=2, num_neighbor=32, drop_rate=0):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_neighbor = num_neighbor
        self.dp = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,q,k,v,attention_mask=None,attention_weight=None,use_top=True):
        # q: bsz, protein_len, hid=heads*hidd'
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)    # q: bsz, heads, protein_len, hid'
        v = self.transpose_for_scores(v)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) # bsz, heads, protein_len, protein_len + bsz, 1, protein_len, protein_len
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if attention_weight is not None:
            attention_weight_sorted_sorted = torch.argsort(torch.argsort(-attention_weight,axis=-1),axis=-1)
            top_mask = (attention_weight_sorted_sorted < self.num_neighbor)
            attention_probs = attention_probs * top_mask
            attention_probs = attention_probs / (torch.sum(attention_probs,dim=-1,keepdim=True) + 1e-5)

        outputs = torch.matmul(attention_probs, v)

        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        outputs = self.dp(outputs)
        outputs = self.ln(outputs)
        return outputs


class GTM(nn.Module):
    def __init__(self, protein_in_dim, protein_out_dim=64, target_dim=1, fc_layer_num=2, atten_layer_num=2, atten_head=2, num_neighbor=32, drop_rate1=0.2619422201258426, drop_rate2=0):
        super().__init__()

        self.input_block = nn.Sequential(
                                         nn.LayerNorm(protein_in_dim, elementwise_affine=True)
                                        ,nn.Linear(protein_in_dim, protein_out_dim)
                                        ,nn.LeakyReLU()
                                        )

        self.hidden_block = []
        for h in range(fc_layer_num-1):
            if h < fc_layer_num-1-1:
                self.hidden_block.extend([
                                          nn.LayerNorm(protein_out_dim, elementwise_affine=True)
                                         ,nn.Dropout(drop_rate1)
                                         ,nn.Linear(protein_out_dim, protein_out_dim)
                                         ,nn.LeakyReLU()
                                         ])
            else:
                self.hidden_block.extend([
                                          nn.LayerNorm(protein_out_dim, elementwise_affine=True)
                                         ,nn.Dropout(drop_rate1)
                                         ,nn.Linear(protein_out_dim, protein_out_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(protein_out_dim, elementwise_affine=True)
                                         ])
        self.hidden_block = nn.Sequential(*self.hidden_block)

        self.layers = nn.ModuleList([Self_Attention(protein_out_dim, atten_head, num_neighbor, drop_rate2) for _ in range(atten_layer_num)])
        self.logit = nn.Linear(protein_out_dim,target_dim)


    def forward(self, protein_node_features, protein_edge_features, protein_dist_matrix, protein_masks):

        protein_embedding = self.input_block(protein_node_features)
        protein_embedding = self.hidden_block(protein_embedding)

        dist_weight = 1.0 / torch.sqrt(1.0+protein_dist_matrix) * protein_masks.unsqueeze(1)
        dist_weight = dist_weight / torch.sum(dist_weight,axis=-1,keepdim=True)

        protein_masks = (1.0-protein_masks) * -10000
        for layer in self.layers:
            protein_embedding = layer(protein_embedding, protein_embedding, protein_embedding, protein_masks.unsqueeze(1).unsqueeze(1), dist_weight.unsqueeze(1)) #.squeeze(1)#,dist_weight.unsqueeze(1)

        y = self.logit(protein_embedding).squeeze(-1) # batch_size * max_len
        return y


