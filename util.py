from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np


def calc_bleu2(hypotheis, refers):
    bleu = 0
    max_bleu = 0
    smoothie = SmoothingFunction()
    for h, r in zip(hypotheis, refers):
        rr = [x for x in r]
        hh = [x for x in h]  # remove UNK
        try:
            hh = hh[: hh.index(3)] # truncated to EOS
        except:
            # print("no eos")
            hh = hh
        cur_bleu = sentence_bleu([rr], hh, weights=(0, 1, 0, 0),
                                 smoothing_function=smoothie.method1)  # BLEU2
        bleu += cur_bleu
        great = None
        great_target = None
        if cur_bleu > max_bleu:
            max_bleu = cur_bleu
            great = hh
            great_target = rr
    return bleu / len(hypotheis), max_bleu, great, great_target


def translate(idx, idx2word):
    word = []
    for w in idx:
        if w != 0:  # pad
            if idx2word[w] == '<EOS>':
                break
            elif idx2word[w] != '<UNK>':
                word.append(idx2word[w])

    return " ".join(word)


def translate_pairs(topic_list, target_list, generated_list, word2idx):
    idx2word = {v: k for k, v in word2idx.items()}
    ret = []
    for t, r, g in zip(topic_list, target_list, generated_list):
        ret.append((translate(t, idx2word), translate(r, idx2word), translate(g, idx2word)))

    return ret



def load_npy(data_config):
    ret = []
    # print(data_config)
    for item in data_config:
        # print(item)
        ret.append(np.load(item))
    return ret

if __name__ == '__main__':
    pass 