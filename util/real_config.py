from ml_collections import config_dict

initial_dictionary = {
    'project_name': 'EvUnroll',
    'model': {
        'depth': 3,
        'base_chs': 32,
        'lstm_input_chs': 16,
        'lstm_hidden_chs': 32,
    },
    'test': {
        'batch_size': 1,
        'workers': 8,
        'fps':20.79,
        'model_path': './trained_model/EvUnroll.pth',
        'result_path': 'results/real_data_nodeblur',
    },
    'test_dataset':{
        'data_root': '/home/zhouxinyu/dataset/EvUnroll/evunroll_realdata',
        'voxel_grid_channel':16,
        'mode':'test',
        'gt_fps': 5000,
        'interval_length': 100,
    },
}

cfg = config_dict.ConfigDict(initial_dictionary)
cfg.model.voxel_grid_channel = cfg.test_dataset.voxel_grid_channel
