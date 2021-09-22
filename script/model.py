import torch, random, os
import torch.nn as nn
import numpy as np


config = {
    'hidden_unit': 64,
    'fc_layer': 2,
    'self_atten_layer': 2,
    'attention_heads': 4,
    'num_neighbor': 30,
    'fc_dropout': 0.2,
    'attention_dropout': 0,
    'class_num': 1
}


class Self_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=4, num_neighbor=30, drop_rate=0):
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
    def __init__(self, protein_in_dim, protein_out_dim=64, target_dim=1, fc_layer_num=2, atten_layer_num=2, atten_head=4, num_neighbor=30, drop_rate1=0.2, drop_rate2=0):
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

        y = self.logit(protein_embedding).squeeze(-1) # batch_size * L
        return y


def Seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_features(data_path, ID, msa):
    if msa == "both" or msa == "single":
        AF2_single = np.load(data_path + f'{ID}_single_norm.npy')
    if msa == "both" or msa == "evo":
        pssm = np.load(data_path + f'{ID}_pssm.npy')
        hhm = np.load(data_path + f'{ID}_hhm.npy')
    dssp = np.load(data_path + f'{ID}_dssp.npy')
    if msa == "both":
        node_features = np.hstack([AF2_single, pssm, hhm, dssp])
    elif msa == "single":
        node_features = np.hstack([AF2_single, dssp])
    elif msa == "evo":
        node_features = np.hstack([pssm, hhm, dssp])

    dismap = np.load(data_path + f'{ID}_dismap.npy')

    node_features = torch.tensor(node_features, dtype = torch.float).unsqueeze(0)
    dismap = torch.tensor(dismap, dtype = torch.float).unsqueeze(0)

    masks = np.ones(node_features.shape[1])
    masks = torch.tensor(masks, dtype = torch.long).unsqueeze(0)

    return node_features, dismap, masks


def predict_one_protein(data_path, model_path, ID, msa):
    Seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_unit = config['hidden_unit']
    fc_layer = config['fc_layer']
    self_atten_layer = config['self_atten_layer']
    attention_heads = config['attention_heads']
    num_neighbor = config['num_neighbor']
    fc_dropout = config['fc_dropout']
    attention_dropout = config['attention_dropout']
    if msa == "both":
        node_dim =  384 + 40 + 14 # Single + Evo + DSSP
    elif msa == "single":
        node_dim =  384 + 14 # Single + DSSP
    elif msa == "evo":
        node_dim = 40 + 14 # Evo + DSSP
    class_num = config['class_num']

    models = []
    for fold in range(5):
        model = GTM(node_dim, hidden_unit, class_num, fc_layer, self_atten_layer, attention_heads, num_neighbor, fc_dropout, attention_dropout)
        if torch.cuda.is_available():
            model.cuda()

        state_dict = torch.load(model_path + 'fold%s.ckpt'%fold, device)
        model.load_state_dict(state_dict)

        model.eval()
        models.append(model)

    with torch.no_grad():
        node_features, dismap, masks = prepare_features(data_path, ID, msa)
        if torch.cuda.is_available():
            node_features = node_features.cuda()
            dismap = dismap.cuda()
            masks = masks.cuda()

        outputs = [model(node_features, None, dismap, masks).sigmoid() for model in models]

        outputs = torch.stack(outputs, 0).mean(0) # Average predictions from 5 models; final shape = (1, L)
        outputs = outputs.detach().cpu().numpy()[0]
    return outputs
