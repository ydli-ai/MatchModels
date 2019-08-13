## CNN-DSSM

传统的CNN - DSSM 只用了一个卷积作为表示层，这里用了类似 TextCNN 的架构作为表示层，用了 kernel_size 不同的 6 个卷积层。



![cnn-dssm](images/cnn-dssm.png)

### 参考

http://research.microsoft.com/pubs/226585/cikm2014_cdssm_final.pdf

https://github.com/airalcorn2/Deep-Semantic-Similarity-Model/blob/master/deep_semantic_similarity_keras.py



## LSTM-DSSM

LSTM-DSSM 用 Bi-LSTM + Attention 作为 DSSM 的表示层。

![lstm-dssm](images/lstm-dssm.png)



## mvlstm

通过 Bi-LSTM 构造对齐矩阵，匹配句子之间的关系。

### ![mvlstm](images/mvlstm.png)

### 参考

https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/mvlstm.py

## ARC II

![arcii](images/arcii.png)

### 参考

https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arcii.py