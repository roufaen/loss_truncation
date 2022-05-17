class Config:
    # learning rate
    start_lr = 5e-4
    max_grad_norm = 1.0

    # model
    max_source_len = 150
    max_target_len = 35
    percentile = 0.2

    # training
    # train_size = 10000
    # valid_size = 500
    train_size = 1e9
    valid_size = 1e9
    batch_size = 64
    save_steps = 2000
    train_log_steps = 1000
    validation_log_steps = 1000
    loss_truncation = False

    # directory
    base_dir = '/data/disk2/private/roufaen/'
    output_dir = base_dir + 'loss_truncation/outputs_no/'
    save_model_dir = base_dir + 'loss_truncation/saved_models_no/'
    model_path = base_dir + 'models/cpm1-small/'
    data_path = base_dir + 'datasets/lcsts/'
