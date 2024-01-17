class Config:
    learning_rate = 2e-5
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 1000
    batch_size = 16
    num_worker = 2
    num_train_epochs = 10
    gradient_accumulation_steps = 1
    sample_rate = 16000
