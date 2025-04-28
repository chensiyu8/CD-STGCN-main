from STGCN import MMSA_run

# import torch.autograd as autograd
# autograd.set_detect_anomaly(True)


MMSA_run(model_name='stgcn', dataset_name='mosei',config_file='./STGCN/config/config_best1.json',seeds=[1111,1111,1111,1111,1111],
         is_tune=False, model_save_dir='./model_mosei/model_saved/saved_model',res_save_dir='./res',
         log_dir='./log_mosei',num_workers=0, gpu_ids=[0])
