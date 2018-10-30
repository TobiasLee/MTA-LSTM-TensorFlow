from tav import TAV
import tensorflow as tf
from config import Config
from dataloader import *
from tat import TAT
import numpy as np
import os


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    log_file = "baseline.txt"
    # set random seed for reproduce
    tf.set_random_seed(88)
    np.random.seed(88)

    config_g = Config().generator_config
    training_config = Config().training_config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # load vocab
    vocab_dict = np.load("./data_zhihu/correct_data/word_dict_zhihu.npy").item()
    idx2word = {v: k for k, v in vocab_dict.items()}
    print(len(vocab_dict))
    config_g["vocab_dict"] = vocab_dict
    config_g["pretrain_wv"] = np.load("./data_zhihu/correct_data/wv_tencent.npy")
    assert config_g["embedding_size"] == config_g["pretrain_wv"].shape[1]
    G = TAT(config_g)  # train TAT
    G.build_placeholder()
    G.build_graph()


    # load data
    si, sl, ti, tl, train_mem = load_npy(Config().train_data_path)
    g_pre_dataloader = GenDataLoader(config_g["batch_size"], si, sl, ti, tl, max_len=100, memory=train_mem)

    sess = tf.Session(config=tf_config)

    saver_g = tf.train.Saver()
    g_pre_dataloader.create_batch()
    sess.run(tf.global_variables_initializer())

    total_step = 0
    print("Start training generator")
    for e in range(1, training_config["baseline_epoch"] + 1):
        avg_loss = 0
        for _ in range(g_pre_dataloader.num_batch):
            total_step += 1
            batch = g_pre_dataloader.next_batch()
            pre_g_loss = G.run_pretrain_step(sess, batch)
            avg_loss += pre_g_loss
        log_data = "epoch: %d  average training loss: %.4f" % (e, avg_loss / g_pre_dataloader.num_batch)
        print(log_data)
        with open(log_file, "a+") as f:
            f.write(log_data + "\n")
        if e % 20 == 0:
            saver_g.save(sess, training_config["tat_path"] + "tat-" + str(total_step))

    print("Training finished")
