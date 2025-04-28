from STGCN import MMSA_run

# import torch.autograd as autograd
# autograd.set_detect_anomaly(True)


MMSA_run(model_name='stgcn', dataset_name='mosi',config_file='./STGCN/config/config_best1.json',seeds=[1111,2222,3333,4444,5555],
         is_tune=False, model_save_dir='./model_mosi/model_saved/saved_model',res_save_dir='./res',
         log_dir='./log_mosi',num_workers=0, gpu_ids=[0])
