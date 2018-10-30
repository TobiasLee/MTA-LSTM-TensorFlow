import numpy as np


class GenDataLoader(object):
    def __init__(self, batch_size, source_index, source_len, target_idx, target_len,
                 max_len, source_label=None, memory=None):
        assert len(source_index) == len(target_idx)
        self.batch_size = batch_size
        self.source_idx = source_index
        self.source_len = source_len
        self.target_idx = target_idx
        self.target_len = target_len
        self.max_len = max_len
        self.has_label = False
        if source_label is not None:
            self.has_label = True
            self.source_label = source_label
        if memory is not None:
            self.has_mem = True
            self.memory = memory
        self.num_batch = len(source_index) // batch_size

    def create_batch(self):
        self.si_batch = np.split(self.source_idx[:self.num_batch * self.batch_size], self.num_batch)
        self.sl_batch = np.split(self.source_len[:self.num_batch * self.batch_size], self.num_batch)
        self.tl_batch = np.split(self.target_len[:self.num_batch * self.batch_size], self.num_batch)
        self.ti_batch = np.split(self.target_idx[:self.num_batch * self.batch_size], self.num_batch)
        if self.has_label:
            self.slbl = np.split(self.source_label[:self.num_batch * self.batch_size], self.num_batch)
        if self.has_mem:
            self.smem = np.split(self.memory[:self.num_batch * self.batch_size], self.num_batch)

        self.g_pointer = 0

    def next_batch(self):
        generator_batch = [self.si_batch[self.g_pointer],
                           self.sl_batch[self.g_pointer],
                           self.ti_batch[self.g_pointer],
                           self.tl_batch[self.g_pointer],
                           ]
        if self.has_label:
            generator_batch.append(self.slbl[self.g_pointer])
        if self.has_mem:
            generator_batch.append(self.smem[self.g_pointer])
        self.g_pointer = (self.g_pointer + 1) % self.num_batch
        return generator_batch

    def reset_pointer(self):
        self.g_pointer = 0


def load_npy(data_config):
    ret = []
    # print(data_config)
    for item in data_config:
        # print(item)
        ret.append(np.load(item))
    return ret
