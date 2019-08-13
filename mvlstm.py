queryInput = Input(shape=(30,600))

titleInput = Input(shape=(30,600))

rep_query = Bidirectional(CuDNNLSTM(128,return_sequences=True))(queryInput)
rep_query = Bidirectional(CuDNNLSTM(128,return_sequences=True))(rep_query)

rep_doc = Bidirectional(CuDNNLSTM(128,return_sequences=True))(titleInput)
rep_doc = Bidirectional(CuDNNLSTM(128,return_sequences=True))(rep_doc)

# Top-k matching layer
matching_matrix = Dot(axes=[2, 2], normalize=False)([rep_query, rep_doc])
matching_signals = Reshape((-1,))(matching_matrix)
matching_topk = Lambda(lambda x: K.tf.nn.top_k(x, k=50, sorted=True)[0])(matching_signals)

# Multilayer perceptron layer.
dnn = Dense(256,activation = 'relu')(matching_topk)
dnn = Dense(64,activation = 'relu')(matching_topk)
out = Dense(1,activation = 'sigmoid')(dnn)
model_mvlstm = Model(inputs=[queryInput,titleInput], outputs=out)
model_mvlstm.compile(loss='binary_crossentropy', 
                optimizer='adam',
                metrics=[metrics.mae, metrics.binary_accuracy])
