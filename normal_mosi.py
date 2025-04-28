from STGCN import STGCN_run


STGCN_run(model_name='stgcn', dataset_name='mosi',config_file='./STGCN/config/config_best.json',seeds=[1111],
         is_tune=False, model_save_dir='./model/model_saved/saved_model',res_save_dir='./res',
         log_dir='./log',num_workers=0, gpu_ids=[0])


