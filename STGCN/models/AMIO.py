"""
AMIO -- All Model in One
"""
import torch.nn as nn

from .multiTask import *
from .singleTask import *
from .missingTask import *
from .ablation import *
from .subNets import AlignSubNet



class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.MODEL_MAP = {
            # single-task
            'stgcn':STGCN,
            # multi-task
            'mstgcn':MSTGCN,
            #comparision
            'stgcn_tfn':STGCN_TFN,
            'stgcn_lmf':STGCN_LMF,
            'stgcn_selfmm':STGCN_SELFMM,
            'stgcn_misa': STGCN_MISA,
            #ablation
            'stgcn_wo_stgcn':STGCN_wo_STGCN,
            'stgcn_wo_el': STGCN_wo_EL,
            'stgcn_wo_nl': STGCN_wo_NL,
            'stgcn_wo_cl': STGCN_wo_CL,
            'stgcn_wo_sas': STGCN_wo_SAS,
            #GCN_layes
            'stgcn_1gcn': STGCN_1GCN,
            'stgcn_3gcn': STGCN_3GCN
        }
        self.need_model_aligned = args.get('need_model_aligned', None)
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        if(self.need_model_aligned):
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()
        lastModel = self.MODEL_MAP[args['model_name']]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        if(self.need_model_aligned):
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        return self.Model(text_x, audio_x, video_x, *args, **kwargs)

