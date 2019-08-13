import numpy as np
from keras.layers import Permute
from keras import backend
from keras.layers import *
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


query_lstm1 = Bidirectional(CuDNNLSTM(K, return_sequences=True))(query)

query_lstm2 = Bidirectional(CuDNNLSTM(K, return_sequences=True))(query_lstm1)

# Attention
query_a1 = Permute((2, 1))(query_lstm2)

query_a3 = Dense(10, activation='softmax')(query_a1)

query_a_probs = Permute((2, 1))(query_a3)

query_attention_out = multiply([query_lstm2, query_a_probs])

query_lstm3 = Bidirectional(CuDNNLSTM(K))(query_attention_out)

query_sem = Dense(L, activation = "tanh", input_dim = K)(query_lstm3)



doc_lstm1 = Bidirectional(CuDNNLSTM(K, return_sequences=True))

doc_lstm2 = Bidirectional(CuDNNLSTM(K, return_sequences=True))

doc_a1 = Permute((2, 1))

doc_a2 = Reshape((300, 10))

doc_att_dense = Dense(30, activation='softmax')

doc_a_probs = Permute((2, 1))


doc_lstm3 = Bidirectional(CuDNNLSTM( 150 ))

doc_sem = Dense(L, activation = "tanh", input_dim = K)


# 正样本

pos_doc_lstm1 = doc_lstm1(pos_doc)

pos_doc_lstm2 = doc_lstm2(pos_doc_lstm1)

pos_doc_a1 = Permute((2, 1))(pos_doc_lstm2)

pos_doc_a3 = doc_att_dense(pos_doc_a1)

pos_doc_probs = Permute((2, 1))(pos_doc_a3)

pos_doc_att_out = multiply([pos_doc_lstm2,pos_doc_probs])

pos_doc_lstm3 = doc_lstm3(pos_doc_att_out)

pos_doc_sem = doc_sem(pos_doc_lstm3)

# 负样本

neg_doc_lstm1 = [doc_lstm1(neg_doc) for neg_doc in neg_docs]
neg_doc_lstm2 = [doc_lstm2(neg_doc) for neg_doc in neg_doc_lstm1]

neg_doc_a1 = [Permute((2, 1))(neg_doc) for neg_doc in neg_doc_lstm2]

neg_doc_a3 = [doc_att_dense(neg_doc) for neg_doc in neg_doc_a1]

neg_doc_probs = [Permute((2, 1))(neg_doc) for neg_doc in neg_doc_a3]

neg_doc_att_out = [multiply([lstm,prb]) for lstm,prb in zip(neg_doc_lstm2,neg_doc_probs)]

neg_doc_lstm3 = [doc_lstm3(neg_doc) for neg_doc in neg_doc_att_out]

neg_doc_sems = [doc_sem(neg_doc_lstm_mx) for neg_doc_lstm_mx in neg_doc_lstm3]

R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True) for neg_doc_sem in neg_doc_sems] # See equation (4).

concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
concat_Rs = Reshape((J + 1, 1))(concat_Rs)

weight = np.array([1]).reshape(1, 1, 1)
with_gamma = Convolution1D(1, 1, padding = "same",
                        input_shape = (J + 1, 1),
                        activation = "linear",
                        use_bias = False,
                        weights = [weight])(concat_Rs) 
with_gamma = Reshape((J + 1, ))(with_gamma)

prob = Activation("softmax")(with_gamma) 

model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
model.compile(optimizer = "adam", loss = "categorical_crossentropy",
                metrics=[metrics.mae, metrics.binary_accuracy])


get_R_Q_D_p = backend.function([query, pos_doc], [R_Q_D_p])
get_R_Q_D_ns = backend.function([query] + neg_docs, R_Q_D_ns)