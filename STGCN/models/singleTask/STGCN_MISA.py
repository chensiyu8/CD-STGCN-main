"""
From: https://github.com/declare-lab/MISA
Paper: MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..subNets import BertTextEncoder
from torch.nn import MultiheadAttention
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data, Batch
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class ReverseLayerF(Function):
    """
    Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
    """

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


# let's define a simple model that can deal with multimodal variable length sequence
class STGCN_MISA(nn.Module):
    def __init__(self, config):
        super(STGCN_MISA, self).__init__()
        args = config
        assert config.use_bert == True

        self.config = config
        self.text_size = config.feature_dims[0]
        self.visual_size = config.feature_dims[2]
        self.acoustic_size = config.feature_dims[1]

        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes if config.train_mode == "classification" else 1
        output_dim = self.output_size

        self.dropout_rate = dropout_rate = config.dropout
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

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



        ##########################################
        # mapping modalities to same sized space
        ##########################################

        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=self.text_out, out_features=config.hidden_size))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))


        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v',
                                  nn.Linear(in_features=self.video_hidden, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a',
                                  nn.Linear(in_features=self.audio_hidden, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())

        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1',
                                          nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2',
                                          nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1',
                                         nn.Linear(in_features=config.hidden_size, out_features=4))

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size * 6,
                                                           out_features=self.config.hidden_size * 3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=self.config.hidden_size * 3, out_features=output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2] * 2,))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)


    def alignment(self, text_x, audio_x, video_x):
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

        utterance_text = t_ts
        utterance_audio = a_ts
        utterance_video = v_ts

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)

        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator(
            (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a) / 3.0)

        # For reconstruction
        self.reconstruct()

        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t,
                         self.utt_shared_v, self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        o = self.fusion(h)
        return o, e_t, e_a, e_v, output_text, output_audio, output_video

    def reconstruct(self, ):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_t, utterance_v, utterance_a):

        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def forward(self, text, audio, video):
        output, e_t, e_a, e_v, output_text, output_audio, output_video = self.alignment(text, audio, video)
        tmp = {
            "M": output,
            'et': e_t,
            'ea': e_a,
            'ev': e_v,
            'T': output_text,
            'A': output_audio,
            'V': output_video
        }
        return tmp
