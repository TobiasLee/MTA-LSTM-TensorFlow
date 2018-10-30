# MTA-LSTM in TensorFlow

TensorFlow implementation of paper [Topic-to-Essay Generation with Neural Networks](http://ir.hit.edu.cn/~xcfeng/xiaocheng%20Feng's%20Homepage_files/final-topic-essay-generation.pdf).

## Motivation: 

The origin [implementation](https://github.com/hit-computer/MTA-LSTM) only provides the MTA-LSTM model, and the code is out-of-date.
In this repo, all three models in the paper are implemented in TensorFlow. And the latest TensorFlow seq2seq and RNNWrapper apis are utilized to make the code clean. 

**If you find this repo is helpful, a star would be so nice!**

## Prerequisites
- Python3
- TensorFlow >= 1.7

## Implementation Notes
1. TAV-LSTM: feed topic embedding average to a forward network to obtain initial state of decoder.
2. TAT-LSTM: at each time step, compute attention on the topic embedding using Bahdanau Attention, and concat with input to the decoder.
3. MTA-LSTM: maintain a **coverage vector** to record the whether topic information has been expressed during the training.

**Note**: beam search used in the origin repo is **NOT SUPPORTED**.

You can refer to the code for more details. If you meet any problems, feel free to raise a issue. 

For the data preprocessing, you may refer to the origin paper and process your own data.
 


## Generated Examples
*TODO* : show some generated examples of three models (in Chinese):

Topic words:
 
TAV-LSTM:

TAT-LSTM:

MTA-LSTM:
