class Config:
    # learning rate
    max_grad_norm = 1.0

    # model
    max_source_len = 128
    max_target_len = 24

    # training
    train_size = 100
    valid_size = 5
    # train_size = 1e9
    # valid_size = 1e9
    batch_size = 64
    save_steps = 2000
    train_log_steps = 1000
    validation_log_steps = 1000
    loss_truncation = False

    # directory
    base_dir = '/data/disk2/private/roufaen/'
    output_dir = base_dir + 'loss_truncation/outputs/'
    save_model_dir = base_dir + 'loss_truncation/saved_models/'
    model_path = base_dir + 'models/cpm1-small/'
    data_path = base_dir + 'datasets/lcsts/'
