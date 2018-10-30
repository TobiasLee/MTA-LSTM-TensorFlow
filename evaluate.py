from config import Config
import tensorflow as tf
import numpy as np
from dataloader import GenDataLoader
from util import *
from generator_tat import TAT
from generator_tav import TAV
from train_baseline import load_npy


if __name__ == '__main__':
    config_g = Config().generator_config
    print(config_g["beam_width"])
    # config_g["pretrain_wv"] = np.load("./data_zhihu/origin/zhihu_wv_50002.npy")
    training_config = Config().training_config
    tf_config = tf.ConfigProto()

    tf_config.gpu_options.allow_growth = True
    # load vocab
    vocab_dict = np.load("./data_zhihu/correct_data/word_dict_zhihu.npy").item()
    idx2word = { v: k for k, v in vocab_dict.items()}
    # print(len(vocab_dict))
    config_g["vocab_dict"] = vocab_dict
    # config_g["pretrain_wv"] = np.load("./data_zhihu/new_data/pretrain_wv.npy")
    config_g["pretrain_wv"] = np.load("./data_zhihu/correct_data/wv_tencent.npy")
    # print(vocab_dict["<UNK>"])  # 1
    # print(max(vocab_dict.values()))
    config_g["is_training"] = False
    config_g["batch_size"] = 128

    G = TAT(config_g)
    G.build_placeholder()
    G.build_graph()
    # D = Discriminator(config_d)
    # D.build_graph()

    sess = tf.Session(config=tf_config)
    saver = tf.train.Saver()
    # path = tf.train.latest_checkpoint("./model/best")

    path = tf.train.latest_checkpoint(training_config["tat_path"])
    print(path)
    sess.run(tf.global_variables_initializer())

    # G.restore(sess, saver, "./model/adversarial/adv-epoch-1")
    # G.restore(sess, saver, path)
    si_tst, sl_tst,  ti_tst, tl_tst, tst_mem = load_npy(Config().test_data_path)
    g_test_dataloader = GenDataLoader(config_g["batch_size"], si_tst, sl_tst, ti_tst, tl_tst,
                                      max_len=100, source_label=None, memory=tst_mem)
    g_test_dataloader.create_batch()
    print("start evaluating....")
    # repeat_time = 1
    # result = []
    # for _ in range(repeat_time):
    #     g_test_dataloader.reset_pointer()
    #     test_samples = []
    #     test_label = []
    #     test_target = []
    #     topic_list = []
    #     test_beam = []
    #     for t_n in range():
    test_bleu, topic_list, test_target, test_samples = G.evaluate(sess, g_test_dataloader, idx2word, get_ret=True)
    print(test_bleu)


    # 7.0802093761391935 pretrain
    # best 5.248051496800161 sampling

    # print(topic_list)
    # print(test_target[0])
    # print(test_samples[0])
    # print(len(topic_list))
    # print(len(test_samples))
    # print(len(test_samples))

    # test_bleu = G.evaluate(sess, g_test_dataloader)
    # print(test_bleu)
    # result.append(test_bleu)
    # print(result)
    # print("average bleu :", sum(result) / len(result))
    # test_beam.extend(beam_samples[:, :, -1])
        # print(topic_idx[:1])
        # print(target_idx[:1])
        # print(topic_idx[:1])
    # print(test_samples[:2])
    # # print(test_label)
    # # print(test_samples[:2])
    ret = translate_pairs(topic_list[:10], test_target[:10], test_samples[:10], vocab_dict)
    for k in ret:
        topic, refer, gen = k[0], k[1], k[2]
        print("input topic: ", topic)
        print("reference: ", refer)
        print("generated: ", gen)
        print()

    # test_bleu, max_bleu, best_ret, best_target = calc_bleu2(test_samples, test_target)
    # print("Test Bleu:  %.4f  Best sentence Bleu : % .4f" % (test_bleu * 100, max_bleu * 100))
    #
    # print("Best Result", translate(best_ret, {v: k for k, v in vocab_dict.items()}))
    # print("Best Target", translate(best_target, {v: k for k, v in vocab_dict.items()}))
