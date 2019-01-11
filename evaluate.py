from config import Config
import tensorflow as tf
import numpy as np
from dataloader import GenDataLoader
from util import *
from tat import TAT
from tav import TAV
from util import load_npy


if __name__ == '__main__':
    config_g = Config().generator_config
    training_config = Config().training_config
    tf_config = tf.ConfigProto()

    tf_config.gpu_options.allow_growth = True
    # load vocab
    vocab_dict = np.load("./data_zhihu/correct_data/word_dict_zhihu.npy").item()
    idx2word = { v: k for k, v in vocab_dict.items()}
    config_g["vocab_dict"] = vocab_dict
    config_g["pretrain_wv"] = np.load("./data_zhihu/correct_data/wv_tencent.npy")

    config_g["is_training"] = False
    config_g["batch_size"] = 128

    G = TAT(config_g)
    G.build_placeholder()
    G.build_graph()
    

    sess = tf.Session(config=tf_config)
    saver = tf.train.Saver()
    G.restore(sess, saver, path=config_g["model_path"]) # you may define you own model path in the config.py
    path = tf.train.latest_checkpoint(training_config["tat_path"])
    print(path)
    sess.run(tf.global_variables_initializer())

    si_tst, sl_tst,  ti_tst, tl_tst, tst_mem = load_npy(Config().test_data_path)
    g_test_dataloader = GenDataLoader(config_g["batch_size"], si_tst, sl_tst, ti_tst, tl_tst,
                                      max_len=100, source_label=None, memory=tst_mem)
    g_test_dataloader.create_batch()
    print("start evaluating....")

    test_bleu, topic_list, test_target, test_samples = G.evaluate(sess, g_test_dataloader, idx2word, get_ret=True)
    print(test_bleu)


    # print some results
    ret = translate_pairs(topic_list[:10], test_target[:10], test_samples[:10], vocab_dict)
    for k in ret:
        topic, refer, gen = k[0], k[1], k[2]
        print("input topic: ", topic)
        print("reference: ", refer)
        print("generated: ", gen)
        print()
