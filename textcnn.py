from keras.layers import *
from keras.models import Model
from keras import metrics

# query输入
queryInput = Input(shape=(30,600))

#title输入
titleInput = Input(shape=(30,600))

x = TimeDistributed(Dense(150, activation='relu'))(queryInput)
xlstm = CuDNNLSTM(150, return_sequences=True)(x)
xlstm1 = GlobalMaxPooling1D()(xlstm)
xa = concatenate([xlstm, x])

xconv1 = Convolution1D(filters=100,
                       kernel_size=1,
                       padding='same',
                       activation='relu')(xa)
xconv1 = GlobalMaxPooling1D()(xconv1)

xconv2 = Convolution1D(filters=100,
                       kernel_size=2,
                       padding='same',
                       activation='relu')(xa)
xconv2 = GlobalMaxPooling1D()(xconv2)

xconv3 = Convolution1D(filters=100,
                       kernel_size=3,
                       padding='same',
                       activation='relu')(xa)
xconv3 = GlobalMaxPooling1D()(xconv3)

xconv4 = Convolution1D(filters=100,
                       kernel_size=4,dilation_rate=2,
                       padding='same',
                       activation='relu')(xa)
xconv4 = GlobalMaxPooling1D()(xconv4)

xconv5 = Convolution1D(filters=100,
                       kernel_size=5,dilation_rate=2,
                       padding='same',
                       activation='relu')(xa)
xconv5 = GlobalMaxPooling1D()(xconv5)

xconv6 = Convolution1D(filters=100,
                       kernel_size=6,
                       padding='same',
                       activation='relu')(xa)
xconv6 = GlobalMaxPooling1D()(xconv6)
xgru = CuDNNGRU(300, return_sequences=True)(xa)
x = concatenate([xconv1,xconv2,xconv3,xconv4,xconv5,xconv6,xlstm1])
x = Dropout(0.2)(x)
x = Dense(200)(x)
x_out = PReLU()(x)


y = TimeDistributed(Dense(150, activation='relu'))(titleInput)
ylstm = CuDNNLSTM(150, return_sequences=True)(y)
ylstm1 = GlobalMaxPooling1D()(ylstm)
ya = concatenate([ylstm, y])

yconv1 = Convolution1D(filters=100,
                       kernel_size=1,
                       padding='same',
                       activation='relu')(ya)
yconv1 = GlobalMaxPooling1D()(yconv1)

yconv2 = Convolution1D(filters=100,
                       kernel_size=2,
                       padding='same',
                       activation='relu')(ya)
yconv2 = GlobalMaxPooling1D()(yconv2)

yconv3 = Convolution1D(filters=100,
                       kernel_size=3,
                       padding='same',
                       activation='relu')(ya)
yconv3 = GlobalMaxPooling1D()(yconv3)

yconv4 = Convolution1D(filters=100,
                       kernel_size=4,dilation_rate=2,
                       padding='same',
                       activation='relu')(ya)
yconv4 = GlobalMaxPooling1D()(yconv4)

yconv5 = Convolution1D(filters=100,
                       kernel_size=5,dilation_rate=2,
                       padding='same',
                       activation='relu')(ya)
yconv5 = GlobalMaxPooling1D()(yconv5)

yconv6 = Convolution1D(filters=100,
                       kernel_size=6,
                       padding='same',
                       activation='relu')(ya)
yconv6 = GlobalMaxPooling1D()(yconv6)
ygru = CuDNNGRU(300, return_sequences=True)(ya)
y = concatenate([yconv1,yconv2,yconv3,yconv4,yconv5,yconv6,ylstm1])
y = Dropout(0.2)(y)
y = Dense(200)(y)
y_out = PReLU()(y)

# interaction
x1,l,lc = [x_out,xlstm,xgru]

x2,r,rc = [y_out,ylstm,ygru]

cross1 = Dot(axes=[2, 2], normalize=True)([l,r])
cross1 = Reshape((-1, ))(cross1)
cross1 = Dropout(0.5)(cross1)
cross1 = Dense(200)(cross1)
cross1 = PReLU()(cross1)

cross2 = Dot(axes=[2, 2], normalize=True)([lc,rc])
cross2 = Reshape((-1, ))(cross2)
cross2 = Dropout(0.5)(cross2)
cross2 = Dense(200)(cross2)
cross2 = PReLU()(cross2)

diff = subtract([x1,x2])
mul = multiply([x1,x2])
x = concatenate([x1,x2,diff,mul,cross1,cross2])
x = BatchNormalization()(x)

x = Dense(500)(x)
x = PReLU()(x)
x = Dropout(0.2)(x)


hidden1 = Dense(200)(x)
hidden1 = PReLU()(hidden1)
hidden1 = Dropout(0.2)(hidden1)


hidden2 = Dense(50)(hidden1)
hidden2 = PReLU()(hidden2)
hidden2 = Dropout(0.2)(hidden2)

out = Dense(1, activation='sigmoid')(hidden2)
model_t2 = Model(inputs=[queryInput,titleInput], outputs=out)
model_t2.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=[metrics.mae, metrics.binary_accuracy])