# config.py
CONFIG = {
    # 경로
    'data_root'  : '/root/data_with_weight_file/dataset',
    'save_dir'   : '/root/data_with_weight_file/checkpoints',
    
    # 모델
    'num_frames' : 16,
    'num_classes': 5,
    
    # 학습
    'batch_size' : 8,
    'epochs'     : 50,
    'lr'         : 1e-4,
    'num_workers': 4,
}