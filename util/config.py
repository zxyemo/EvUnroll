from ml_collections import config_dict

initial_dictionary = {
    'project_name': 'EvUnroll',
    'model': {
        'depth': 3,
        'base_chs': 32,
        'lstm_input_chs': 16,
        'lstm_hidden_chs': 32,
    },
    'train': {
        'batch_size': 8,
        'val_batch_size': 4,
        'max_epoch': 50,
        'lr': 2e-4,
        'workers': 12,
        'ckp_path': 'checkpoints/fusion',
        'resume_path': None,
        'log_path': 'logs/fusion',
        'val_freq': 2,
        'log_freq': 10,
        'checkpoint_freq': 2,
        'perceptual_loss_weight': 0.1,
        'flow_loss_weight': 0.2,
    },
    'test': {
        'batch_size': 4,
        'workers': 8,
        'model_path': './trained_model/EvUnroll.pth',
        'result_path': 'results/test',
    },
    'train_dataset':{
        'img_root': '/media/zhouxinyu/DATA/data/EvUnroll/Gev-RS-360/train', # path to the training dataset
        'event_root': '/home/zhouxinyu/dataset/EvUnroll/Gev-RS-DVS', # path to the event files
        'gt_root': '/mnt/nas1home/dell_D/data/VEO_image_all', # path to the ground truth images
        'crop_size': [256, 256],
        'voxel_grid_channel':16,
        'mode':'train',
        'gt_fps': 5000,
        'interval_length': 100,
    },
    'test_dataset':{
        'img_root': '/media/zhouxinyu/DATA/data/EvUnroll/Gev-RS-360/test', # path to the testing dataset
        'event_root': '/home/zhouxinyu/dataset/EvUnroll/Gev-RS-DVS', # path to the event files
        'gt_root': '/mnt/nas1home/dell_D/data/VEO_image_all', # path to the ground truth images
        'crop_size': [None, None],
        'voxel_grid_channel':16,
        'mode':'test',
        'gt_fps': 5000,
        'interval_length': 100,
    },

}

cfg = config_dict.ConfigDict(initial_dictionary)
cfg.model.voxel_grid_channel = cfg.train_dataset.voxel_grid_channel
