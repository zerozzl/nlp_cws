# 自然语言处理-分词
对比常见分词模型效果，主要涉及以下三种模型：
- [Convolutional Neural Network with Word Embeddings for Chinese Word Segmentation](https://arxiv.org/abs/1711.04411)
- [State-of-the-art Chinese Word Segmentation with Bi-LSTMs](http://arxiv.org/pdf/1808.06511)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## 分词效果
### CONV-SEG
|-|Simple|CRF|Word Feature|Fix Embedding|CRF + Word Feature|CRF + Fix Embedding|Word Feature + Fix Embedding|CRF + Word Feature + Fix Embedding|
|----|----|----|----|----|----|----|----|----|
|PKU|0.954|0.953|<b>0.958</b>|0.951|<b>0.958</b>|0.951|0.957|0.957|
|MSR|0.969|0.971|<b>0.977</b>|0.965|<b>0.977</b>|0.966|0.976|<b>0.977</b>|

### BiLSTM
|-|Simple|Word Feature|Stack LSTM|CRF|CRF + Stack LSTM|Stack LSTM + Word Feature|CRF + Stack LSTM + Word Feature|CRF + Stack LSTM + Fix Embedding|CRF + Stack LSTM + Word Feature + Fix Embedding|
|----|----|----|----|----|----|----|----|----|----|
|PKU|0.867|0.946|0.881|0.923|0.928|0.947|<b>0.953</b>|0.913|0.943|
|MSR|0.845|0.941|0.856|0.89|0.902|0.946|<b>0.959</b>|0.869|0.934|

### Bert
|-|Simple|CRF|Fix Embedding|
|----|----|----|----|
|PKU|<b>0.967</b>|0.965|0.866|
|MSR|0.981|<b>0.982</b>|0.871|
