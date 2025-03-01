# configs.py
class Configs:
    num_epochs = 500
    batch_size = 32
    learning_rate = 0.0001
    num_classes = 8
    base_channels = 64
    base_channels_multiples = (1, 2, 4, 8)
    apply_attention = (False, False, True, False)
    dropout_rate = 0.2
    time_multiple = 2
    d_model = 32
    input_channels = 3
    output_channels = 8
    num_timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    checkpoint_path = 'checkpoint_DGF_zong_cs.pth'
    best_model_path = 'best_model_DGF_zong.pth'
    data_root = './data/DZong'
    train_dir = f'{data_root}/train'
    val_dir = f'{data_root}/val'
    test_dir = f'{data_root}/test'