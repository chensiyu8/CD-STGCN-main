import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import MultiheadAttention
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data, Batch
from ..subNets import BertTextEncoder




# bilstm for multitask
class SubNet(nn.Module):
    '''
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(SubNet, self).__init__()
        if num_layers == 1:
            dropout = 0
        nums = 1
        if bidirectional == True:
            nums = nums * 2
        self.nums = nums * num_layers  # 几个LSTM堆叠
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(self.nums * hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        if self.num_layers == 1 and self.bidirectional == False:
            _, final_states = self.rnn(x)
            h = self.dropout(final_states[0].squeeze())
            y_1 = torch.relu(self.linear_1(h))
        else:
            _, final_states = self.rnn(x)# final_state[0] : (numlayers*direction , batchsize, hiddensize)
            x1 = final_states[0].permute(1, 0, 2) # numlayers = 1 , x1: (B, D ,H) ,final_state始终存有前向或后向最后一个time step的hidden state
            batchsize = x1.shape[0]
            x1 = x1.reshape(batchsize, -1)
            x1 = self.dropout(x1)
            y_1 = torch.relu(self.linear_1(x1))
        return y_1



## fusion module
class Fusionnet(nn.Module):
    def __init__(self, in_dim, nheads, fusion_dim, outdim, layers):
        super(Fusionnet, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, batch_first=True)
        self.fusion_att_layer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        #self.fusion_att_layer = MultiheadAttention(embed_dim=in_dim, num_heads=nheads, batch_first=True)
        self.linear1 = nn.Linear(3 * in_dim, fusion_dim)
        self.linear2 = nn.Linear(fusion_dim, fusion_dim)
        self.linear3 = nn.Linear(fusion_dim, outdim)

    def forward(self, x):
        #x1,_ = self.fusion_att_layer(x, x, x)  # x :(batchsize, 3 , d)
        x1 = self.fusion_att_layer(x)  # x :(batchsize, 3 , d)
        x1 = torch.cat([x1[:, 0], x1[:, 1], x1[:, 2]], dim=1)  # (batch, 3d)
        x2 = F.relu(self.linear1(x1))
        x3 = F.relu(self.linear2(x2))
        y = self.linear3(x3)
        return y, x1







## model++
class STGCN_wo_STGCN(nn.Module):
    def __init__(self, args):
        super(STGCN_wo_STGCN, self).__init__()
        self.bert_text_dim = 768
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.text_out = args.text_out

        self.audio_prob, self.video_prob, self.text_prob = args.dropouts
        self.post_text_prob, self.post_audio_prob, self.post_video_prob, self.post_fusion_prob = args.post_dropouts

        self.post_fusion_dim = args.post_fusion_dim
        self.post_text_dim = args.post_text_dim
        self.post_audio_dim = args.post_audio_dim
        self.post_video_dim = args.post_video_dim

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)

        self.text_bilstm = SubNet(self.bert_text_dim, self.text_hidden, self.text_out, dropout=self.text_prob)
        self.audio_bilstm = SubNet(self.audio_in, self.audio_hidden, self.audio_hidden, dropout=self.audio_prob)
        self.video_bilstm = SubNet(self.video_in, self.video_hidden, self.video_hidden, dropout=self.video_prob)
        ####

        # define the post_fusion layers
        self.text_fusion_layer_1 = nn.Linear(self.text_out, self.post_fusion_dim)
        # self.text_fusion_layer_1 = nn.Linear(768, self.text_out)
        self.audio_fusion_layer_1 = nn.Linear(self.audio_hidden, self.post_fusion_dim)
        self.video_fusion_layer_1 = nn.Linear(self.video_hidden, self.post_fusion_dim)
        self.post_fusion_layer = Fusionnet(self.post_fusion_dim, args.nheads, self.post_fusion_dim, output_dim, args.layers)


    def forward(self, text_x, audio_x, video_x):
        # bert
        text_x, text_cls = self.text_model(text_x) # 取出非CLS和SEP的部分

        #filter_attention
        text_h = self.text_bilstm(text_x)
        audio_h =self.audio_bilstm(audio_x)
        video_h = self.video_bilstm(video_x)

        # fusion
        mt = self.text_fusion_layer_1(text_h)
        ma = self.audio_fusion_layer_1(audio_h)
        mv = self.video_fusion_layer_1(video_h)
        fusion_h = torch.stack((mt, ma, mv), dim=1)  # (batchsize, 3, dim)
        output_fusion, fusion_final = self.post_fusion_layer(fusion_h)

        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_final,
            'M': output_fusion,

        }
        return res







