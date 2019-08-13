import numpy as np
from keras import backend
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model
from keras import metrics

K = 300 
L = 128 
J = 2 

query = Input(shape = (10, 300))
pos_doc = Input(shape = (30, 300))
neg_docs = [Input(shape = (30, 300)) for j in range(J)]

# 在 DSSM 的表示层使用了类似 TextCNN 的架构

query_conv1 = Convolution1D(K, 1, padding = "same",
                            input_shape = (None, WORD_DEPTH),
                            activation = "tanh")(query) # See equation (2).

query_conv2 = Convolution1D(K, 2, padding = "same",
                            input_shape = (None, WORD_DEPTH),
                            activation = "tanh")(query)

query_conv3 = Convolution1D(K, 3, padding = "same",
                            input_shape = (None, WORD_DEPTH),
                            activation = "tanh")(query)

query_conv4 = Convolution1D(K, 4, padding = "same",dilation_rate=2,
                            input_shape = (None, WORD_DEPTH),
                            
                            activation = "tanh")(query)  

query_conv5 = Convolution1D(K, 5, padding = "same",dilation_rate=2,
                            input_shape = (None, WORD_DEPTH),
                            activation = "tanh")(query)

query_conv6 = Convolution1D(K, 6, padding = "same",
                            input_shape = (None, WORD_DEPTH),
                            activation = "tanh")(query)                                          

# 下一步，将最大池化层应用在卷积后的query上。
# 这一操作选择了每一列的最大值

query_max1 = Lambda(lambda x: backend.max(x, axis = 1),output_shape = (K, ))(query_conv1) 

query_max2 = Lambda(lambda x: backend.max(x, axis = 1),output_shape = (K, ))(query_conv2)

query_max3 = Lambda(lambda x: backend.max(x, axis = 1),output_shape = (K, ))(query_conv3)

query_max4 = Lambda(lambda x: backend.max(x, axis = 1),output_shape = (K, ))(query_conv4)

query_max5 = Lambda(lambda x: backend.max(x, axis = 1),output_shape = (K, ))(query_conv5)

query_max6 = Lambda(lambda x: backend.max(x, axis = 1),output_shape = (K, ))(query_conv6)

query_concat_1_2 = concatenate([query_max1,query_max2])

query_concat_3_4 = concatenate([query_max3,query_max4])

query_concat_5_6 = concatenate([query_max5,query_max6])


query_sem1 = Dense(L, activation = "tanh", input_dim = K*2)(query_concat_1_2)

query_sem2 = Dense(L, activation = "tanh", input_dim = K*2)(query_concat_3_4)

query_sem3 = Dense(L, activation = "tanh", input_dim = K*2)(query_concat_5_6)

# 在这一步中，生成一个句向量来表示一个query。这是一个标准的神经网络层。


query_concat = concatenate([query_sem1,query_sem2,query_sem3])

query_sem = Dense(L, activation = "tanh", input_dim = K*3)(query_concat) 

doc_conv1 = Convolution1D(K, 1, padding = "same",
                            input_shape = (None, WORD_DEPTH),
                            activation = "tanh")
                            
doc_conv2 = Convolution1D(K, 2, padding = "same",
                            input_shape = (None, WORD_DEPTH),
                            activation = "tanh")
     
doc_conv3 = Convolution1D(K, 3, padding = "same",
                            input_shape = (None, WORD_DEPTH),
                            activation = "tanh")
                            
doc_conv4 = Convolution1D(K, 4, padding = "same",dilation_rate=2,
                            input_shape = (None, WORD_DEPTH),
                            activation = "tanh")

doc_conv5 = Convolution1D(K, 5, padding = "same",dilation_rate=2,
                            input_shape = (None, WORD_DEPTH),
                            activation = "tanh")
                            
doc_conv6 = Convolution1D(K, 6, padding = "same",
                            input_shape = (None, WORD_DEPTH),
                            activation = "tanh")                                                        

                                                                                    
doc_max = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (K, ))

doc_sem1 = Dense(L, activation = "tanh", input_dim = K*2)

doc_sem2 = Dense(L, activation = "tanh", input_dim = K*2)

doc_sem3 = Dense(L, activation = "tanh", input_dim = K*2)

doc_sem = Dense(L, activation = "tanh", input_dim = K*3)


# 正样本
pos_doc_conv1 = doc_conv1(pos_doc)
pos_doc_max1 = doc_max(pos_doc_conv1)

pos_doc_conv2 = doc_conv2(pos_doc)
pos_doc_max2 = doc_max(pos_doc_conv2)

pos_doc_conv3 = doc_conv3(pos_doc)
pos_doc_max3 = doc_max(pos_doc_conv3)

pos_doc_conv4 = doc_conv4(pos_doc)
pos_doc_max4 = doc_max(pos_doc_conv4)

pos_doc_conv5 = doc_conv5(pos_doc)
pos_doc_max5 = doc_max(pos_doc_conv5)

pos_doc_conv6 = doc_conv6(pos_doc)
pos_doc_max6 = doc_max(pos_doc_conv6)

pos_doc_concat_1_2 = concatenate([pos_doc_max1,pos_doc_max2])

pos_doc_concat_3_4 = concatenate([pos_doc_max3,pos_doc_max4])

pos_doc_concat_5_6 = concatenate([pos_doc_max5,pos_doc_max6])


pos_doc_sem1 = doc_sem1(pos_doc_concat_1_2)

pos_doc_sem2 = doc_sem2(pos_doc_concat_3_4)

pos_doc_sem3 = doc_sem3(pos_doc_concat_5_6)

pos_doc_concat = concatenate([pos_doc_sem1,pos_doc_sem2,pos_doc_sem3])

pos_doc_sem = doc_sem(pos_doc_concat)


# 负样本

neg_doc_convs1 = [doc_conv1(neg_doc) for neg_doc in neg_docs]
neg_doc_maxes1 = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs1]

neg_doc_convs2 = [doc_conv2(neg_doc) for neg_doc in neg_docs]
neg_doc_maxes2 = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs2]

neg_doc_convs3 = [doc_conv3(neg_doc) for neg_doc in neg_docs]
neg_doc_maxes3 = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs3]

neg_doc_convs4 = [doc_conv4(neg_doc) for neg_doc in neg_docs]
neg_doc_maxes4 = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs4]

neg_doc_convs5 = [doc_conv5(neg_doc) for neg_doc in neg_docs]
neg_doc_maxes5 = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs5]

neg_doc_convs6 = [doc_conv6(neg_doc) for neg_doc in neg_docs]
neg_doc_maxes6 = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs6]

neg_doc_concats_1_2 = [concatenate([l1,l2]) for l1,l2 in zip(neg_doc_maxes1,neg_doc_maxes2)]

neg_doc_concats_3_4 = [concatenate([l3,l4]) for l3,l4 in zip(neg_doc_maxes3,neg_doc_maxes4)]

neg_doc_concats_5_6 = [concatenate([l5,l6]) for l5,l6 in zip(neg_doc_maxes5,neg_doc_maxes6)]

neg_doc_sems1 = [doc_sem1(neg_doc_concat) for neg_doc_concat in neg_doc_concats_1_2]

neg_doc_sems2 = [doc_sem2(neg_doc_concat) for neg_doc_concat in neg_doc_concats_3_4]

neg_doc_sems3 = [doc_sem3(neg_doc_concat) for neg_doc_concat in neg_doc_concats_5_6]

neg_doc_concats = [concatenate([l1,l2,l3]) for l1,l2,l3 in zip(neg_doc_sems1,neg_doc_sems2,neg_doc_sems3)]


neg_doc_sems = [doc_sem(neg_doc_concat) for neg_doc_concat in neg_doc_concats]

# 计算 query 和每个 title 的余弦相似度 R(Q, D)

R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True) for neg_doc_sem in neg_doc_sems] 

concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
concat_Rs = Reshape((J + 1, 1))(concat_Rs)

# 在这一步，将每个 R(Q, D) 乘以 gamma。 
# 在论文中，gamma 是 softmax 的平滑因子，
# 这里用 CNN 来学习gamma的值，是一个 1*1 的卷积核。

weight = np.array([1]).reshape(1, 1, 1)
with_gamma = Convolution1D(1, 1, padding = "same",
                        input_shape = (J + 1, 1),
                        activation = "linear",
                        use_bias = False,
                        weights = [weight])(concat_Rs) 
with_gamma = Reshape((J + 1, ))(with_gamma)

prob = Activation("softmax")(with_gamma) 

model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
model.compile(optimizer = "adadelta", loss = "categorical_crossentropy",
                metrics=[metrics.mae, metrics.binary_accuracy])
                

get_R_Q_D_p = backend.function([query, pos_doc], [R_Q_D_p])
get_R_Q_D_ns = backend.function([query] + neg_docs, R_Q_D_ns)