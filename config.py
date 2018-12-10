class Config(object):
    training_config = {
        "tat_path": "./model/tat/",
        "baseline_epoch": 80,
        "tav_path": "./model/tav/",
        "mta_path": "./model/mta/"
    }

    train_data_path = [
        # please use your own preprocessed data. 
    ]
    test_data_path = [
        # please use your own preprocessed data. 
    ]

    generator_config = {
        "embedding_size": 200,  #
        "hidden_size": 512,  #
        "max_len": 100,
        "start_token": 0,
        "eos_token": 1,
        "batch_size": 128,
        "vocab_size": 50004,
        "grad_norm": 10,
        "topic_num": 5,
        "is_training": True,
        "keep_prob": .5,
        "norm_init": 0.05,
        "normal_std": 1,
        "learning_rate": 1e-3,
        "beam_width": 5,
        "mem_num": 60
    }


