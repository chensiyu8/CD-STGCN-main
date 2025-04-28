"""
paper: Tensor Fusion Network for Multimodal Sentiment Analysis
From: https://github.com/A2Zadeh/TensorFusionNetwork
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import MultiheadAttention
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data, Batch
from ..subNets import BertTextEncoder


class GlobalSelfAttention(nn.Module):
    def __init__(self, in_size, dropout):
        super(GlobalSelfAttention, self).__init__()
        self.linear_1 = nn.Linear(in_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(2 * in_size, 1)
        self.linear_3 = nn.Linear(2 * in_size, in_size)

    def forward(self, x):
        dropped_1 = self.dropout(x)
        att = nn.Sigmoid()(self.linear_1(dropped_1))
        vm = torch.mul(att, x).mean(1)
        vm = vm.unsqueeze(1).repeat(1, x.shape[1], 1)
        vs = torch.cat([x, vm], dim=-1)
        dropped_2 = self.dropout(vs)
        att_new = nn.Sigmoid()(self.linear_2(dropped_2))
        y = torch.mul(att * att_new, vs)
        y_1 = F.relu(self.linear_3(y), inplace=True)
        return y_1


## STGCN module
class G_LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        '''
        G_i=BiLSTM_G(x_i), h_{i-1}=G_{i-1},
        G_i= linear(G正,G反)
        '''
        super(G_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.glinear = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x):
        '''
        :param x: (batch ,seqlen , in_dim)
        :return: out 图状态向量 ，x 原输入
        '''
        out, _ = self.rnn(x)
        out = self.glinear(out)
        return out, x


class unimodal_edge_attention(nn.Module):
    def __init__(self, in_dim, g_dim, hidden_dim, seq_len, heads):
        super(unimodal_edge_attention, self).__init__()
        self.in_dim = in_dim
        self.g_dim = g_dim
        self.hidden_dim = hidden_dim
        self.st_in_dim = 2 * g_dim
        self.seqlen = seq_len
        self.linear = nn.Linear(in_dim, g_dim)
        self.edge_attention = MultiheadAttention(embed_dim=self.st_in_dim, num_heads=heads, batch_first=True)

    def forward(self, x, g):
        '''
        :param x: (batchsize, seqlen, in_dim) g 图状态向量
        :return: 图batch结构数据
        '''
        x1 = self.linear(x)
        x1 = torch.cat([x1, g], dim=2)
        data_list = []
        lu, batch_edge_weight = self.edge_attention(x1, x1, x1)
        batch_edge_weight = torch.relu(batch_edge_weight)
        for i in range(x.shape[0]):
            S = batch_edge_weight[i]
            edge_index = S.nonzero().t()
            edge_weight = S[S.nonzero()[:, 0], S.nonzero()[:, 1]]
            q = Data(x=x[i], edge_index=edge_index, edge_weight=edge_weight)
            data_list.append(q)
        batch = Batch.from_data_list(data_list)
        return batch, lu


class GCNsubnet(nn.Module):
    def __init__(self, in_dim, g_dim, hidden_dim, seq_len, dropout=0.2, heads=4):
        super(GCNsubnet, self).__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.BatchNorm1d(in_dim)
        self.g_rnn = G_LSTM(in_dim, g_dim)
        self.edge_compute = unimodal_edge_attention(in_dim, g_dim, hidden_dim, seq_len, heads)
        self.GNN = Sequential('x, edge_index, edge_weight', [
            (GCNConv(in_dim, 2 * hidden_dim), 'x, edge_index, edge_weight -> x1'),
            (GCNConv(2 * hidden_dim, hidden_dim), 'x1, edge_index, edge_weight -> x2'),
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_flat = x.view(-1, x.shape[2])
        x0 = self.norm(x_flat)
        normalized_x = x0.view(x.shape[0], x.shape[1], x.shape[2])
        g, x1 = self.g_rnn(normalized_x)
        batch, lu = self.edge_compute(x1, g)
        f = batch.x
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        output = self.GNN(f, edge_index, edge_weight)
        output = output.view(x.shape[0], x.shape[1], self.hidden_dim)
        res = self.dropout(output)
        return res, lu


## BiLSTM module for stgcn
class SubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        LSTM output 格式参考https://zhuanlan.zhihu.com/p/39191116
        '''
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
        output:
            y: (batch_size, sequence_len, out_size)
        '''
        if self.num_layers == 1 and self.bidirectional == False:
            y, final_states = self.rnn(x)
            h = self.dropout(y)
            y_1 = torch.relu(self.linear_1(h))
        else:
            y, final_states = self.rnn(x)  # final_state[0] : (numlayers*direction , batchsize, hiddensize)
            x1 = self.dropout(y)
            y_1 = torch.relu(self.linear_1(x1))
        return y_1


# bilstm for multitask
class SubNet_mt(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(SubNet_mt, self).__init__()
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
            _, final_states = self.rnn(x)  # final_state[0] : (numlayers*direction , batchsize, hiddensize)
            x1 = final_states[0].permute(1, 0,
                                         2)  # numlayers = 1 , x1: (B, D ,H) ,final_state始终存有前向或后向最后一个time step的hidden state
            batchsize = x1.shape[0]
            x1 = x1.reshape(batchsize, -1)
            x1 = self.dropout(x1)
            y_1 = torch.relu(self.linear_1(x1))
        return y_1


class STGCN_TFN(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    '''

    def __init__(self, args):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, similar to input_dims
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(STGCN_TFN, self).__init__()
        self.bert_text_dim = 768
        # dimensions are specified in the order of audio, video and text
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        output_dim = args.num_classes if args.train_mode == "classification" else 1
        self.output_dim =output_dim
        self.text_out = args.text_out
        self.post_fusion_dim = args.post_fusion_dim
        self.post_text_dim = args.post_text_dim
        self.post_audio_dim = args.post_audio_dim
        self.post_video_dim = args.post_video_dim
        self.audio_prob, self.video_prob, self.text_prob = args.dropouts
        self.post_text_prob, self.post_audio_prob, self.post_video_prob, self.post_fusion_prob = args.post_dropouts

        # use Stgcn replace
        self.text_att = GlobalSelfAttention(self.bert_text_dim, dropout=self.text_prob)
        self.video_att = GlobalSelfAttention(self.video_in, dropout=self.video_prob)
        self.audio_att = GlobalSelfAttention(self.audio_in, dropout=self.audio_prob)
        self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)



        ####
        self.T_GCN = GCNsubnet(self.bert_text_dim, self.text_hidden, self.text_hidden, args.seq_lens[0]-2,#序列去除了CLS和SEP
                               dropout=self.text_prob, heads=args.nheads)
        self.T_att = MultiheadAttention(embed_dim=self.text_hidden, num_heads=args.nheads, batch_first=True)
        self.T_combine_cls_seq_layer = nn.Linear(self.text_hidden+self.bert_text_dim, self.text_out)
        self.T_layer1 = nn.Linear(self.text_out, self.text_hidden)
        self.T_layer2 = nn.Linear(self.text_hidden, self.post_text_dim)
        self.T_layer3 = nn.Linear(self.post_text_dim, output_dim)

        self.A_GCN = GCNsubnet(self.audio_in, self.audio_hidden, self.audio_hidden, args.seq_lens[1], dropout=self.audio_prob, heads= args.nheads )
        self.A_att = MultiheadAttention(embed_dim=self.audio_hidden, num_heads=args.nheads, batch_first=True)
        self.A_layer1 = nn.Linear(self.audio_hidden, self.audio_hidden)
        self.A_layer2 = nn.Linear(self.audio_hidden, self.post_audio_dim)
        self.A_layer3 = nn.Linear(self.post_audio_dim, output_dim)

        self.V_GCN = GCNsubnet(self.video_in, self.video_hidden, self.video_hidden, args.seq_lens[2], dropout=self.video_prob, heads= args.nheads )
        self.V_att = MultiheadAttention(embed_dim=self.video_hidden, num_heads=args.nheads, batch_first=True)
        self.V_layer1 = nn.Linear(self.video_hidden, self.video_hidden)
        self.V_layer2 = nn.Linear(self.video_hidden, self.post_video_dim)
        self.V_layer3 = nn.Linear(self.post_video_dim, output_dim)

        ###


        # multitask for text_edge
        self.post_text_dropout = nn.Dropout(p=self.post_text_prob)
        self.t_bilstm = SubNet_mt(2 * self.text_hidden, self.text_hidden, self.text_hidden, dropout=self.audio_prob,
                                 bidirectional=True)
        self.post_text_layer_1 = nn.Linear(self.text_hidden, self.post_text_dim)
        self.post_text_layer_2 = nn.Linear(self.post_text_dim, self.post_text_dim)
        self.post_text_layer_3 = nn.Linear(self.post_text_dim, output_dim)

        # multitask for audio_edge
        self.post_audio_dropout = nn.Dropout(p=self.post_audio_prob)
        self.a_bilstm = SubNet_mt(2*self.audio_hidden, self.audio_hidden, self.audio_hidden, dropout=self.audio_prob,
               bidirectional=True)
        self.post_audio_layer_1 = nn.Linear(self.audio_hidden, self.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(self.post_audio_dim, self.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(self.post_audio_dim, output_dim)

        # multitask for video_edge
        self.post_video_dropout = nn.Dropout(p=self.post_video_prob)
        self.v_bilstm = SubNet_mt(2*self.video_hidden, self.video_hidden, self.video_hidden, dropout=self.audio_prob,
                                   bidirectional=True)
        self.post_video_layer_1 = nn.Linear(self.video_hidden, self.post_video_dim)
        self.post_video_layer_2 = nn.Linear(self.post_video_dim, self.post_video_dim)
        self.post_video_layer_3 = nn.Linear(self.post_video_dim, output_dim)


        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1),
                                             self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, self.output_dim)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, text_x, audio_x, video_x):
        ###STGCN
        # bert
        text_x, text_cls = self.text_model(text_x)  # 取出非CLS和SEP的部分

        # filter_attention
        video_selfatt = self.video_att(video_x)
        audio_selfatt = self.audio_att(audio_x)
        text_selfatt = self.text_att(text_x)

        # stgcn{t,a,v}
        tH, lt = self.T_GCN(text_selfatt)
        aH, la = self.A_GCN(audio_selfatt)
        vH, lv = self.V_GCN(video_selfatt)
        # multi-task -> node_select

        t_ts, _ = self.T_att(tH, tH, tH)
        t_ts = torch.mean(t_ts, dim=1)
        t_ts = self.T_combine_cls_seq_layer(torch.cat([t_ts, text_cls], dim=-1))
        # text_node
        c_t = self.T_layer1(t_ts)
        c_t = self.T_layer2(c_t)
        output_text = self.T_layer3(c_t)

        a_ts, _ = self.A_att(aH, aH, aH)
        a_ts = torch.mean(a_ts, dim=1)
        # audio_node
        c_a = self.A_layer1(a_ts)
        c_a = self.A_layer2(c_a)
        output_audio = self.A_layer3(c_a)

        v_ts, _ = self.V_att(vH, vH, vH)
        v_ts = torch.mean(v_ts, dim=1)
        # video_node
        c_v = self.V_layer1(v_ts)
        c_v = self.V_layer2(c_v)
        output_video = self.V_layer3(c_v)

        # text
        x_t = self.post_text_dropout(lt)
        x_t = self.t_bilstm(x_t)
        x_t = F.relu(self.post_text_layer_1(x_t), inplace=True)
        x_t = F.relu(self.post_text_layer_2(x_t), inplace=True)
        e_t = self.post_text_layer_3(x_t)
        # audio_edge
        x_a = self.post_audio_dropout(la)
        x_a = self.a_bilstm(x_a)
        x_a = F.relu(self.post_audio_layer_1(x_a), inplace=True)
        x_a = F.relu(self.post_audio_layer_2(x_a), inplace=True)
        e_a = self.post_audio_layer_3(x_a)
        # video_edge
        x_v = self.post_video_dropout(lv)
        x_v = self.v_bilstm(x_v)
        x_v = F.relu(self.post_video_layer_1(x_v), inplace=True)
        x_v = F.relu(self.post_video_layer_2(x_v), inplace=True)
        e_v = self.post_video_layer_3(x_v)
        ###STGCN

        text_h = t_ts
        audio_h = a_ts
        video_h = v_ts
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        add_one = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(audio_h).to(text_x.device)
        _audio_h = torch.cat((add_one, audio_h), dim=1)
        _video_h = torch.cat((add_one, video_h), dim=1)
        _text_h = torch.cat((add_one, text_h), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))

        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped), inplace=True)
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1), inplace=True)
        output = self.post_fusion_layer_3(post_fusion_y_2)
        if self.output_dim == 1:  # regression
            output = torch.sigmoid(output)
            output = output * self.output_range + self.output_shift

        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'et': e_t,
            'ea': e_a,
            'ev': e_v,
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_f': fusion_tensor,
            'M': output
        }

        return res
