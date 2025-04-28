"""
ATIO -- All Trains in One
"""
from .multiTask import *
from .singleTask import *
from .ablation import *
from .missingTask import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # single-task
            'stgcn': STGCN,
            # multi-task
            'mstgcn': MSTGCN,
            #comparision
            'stgcn_tfn': STGCN_TFN,
            'stgcn_lmf': STGCN_LMF,
            'stgcn_selfmm': STGCN_SELFMM,
            'stgcn_misa':STGCN_MISA,
            #ablation
            'stgcn_wo_stgcn': STGCN_wo_STGCN,
            'stgcn_wo_el': STGCN_wo_EL,
            'stgcn_wo_nl': STGCN_wo_NL,
            'stgcn_wo_cl': STGCN_wo_CL,
            'stgcn_wo_sas':STGCN_wo_SAS,
            #GCN_layers
            'stgcn_1gcn': STGCN_1GCN,
            'stgcn_3gcn': STGCN_3GCN
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args['model_name']](args)
