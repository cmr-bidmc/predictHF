from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization

def build_model_ae(input_dim, model_layers):
    model = Sequential()
    indim = input_dim
    my_activation = 'sigmoid'
    my_loss = 'binary_crossentropy'

    for li in range(model_layers.size): # Encoder Network
        model.add(Dense(model_layers[li], input_dim=indim, activation=my_activation,name='enc_dns_'+str(li)))
        model.add(BatchNormalization(name='enc_bn_'+str(li)))
        model.add(Dropout(0.05, name='enc_dp_'+str(li)))
        indim = model_layers[li]

    for li in range(model_layers.size-1): # Decoder Network
        model.add(Dense(model_layers[model_layers.size-li-2], input_dim=indim, activation=my_activation,name='dec_dns_'+str(li)))
        model.add(BatchNormalization(name='dec_bn_'+str(li)))
        model.add(Dropout(0.00025,name='dec_dp_'+str(li)))
        indim = model_layers[li]  # the output become input for the next layer

    model.add(Dense(input_dim, input_dim=indim, activation='sigmoid',name='dec_dns_out'))

    model.compile(loss=my_loss, optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def build_model_predict(input_dim, model_layers_ae,model_layers_prd):
    model = Sequential()
    indim = input_dim
    my_activation = 'sigmoid'
    for li in range(model_layers_ae.size): # Encoder Network
        model.add(Dense(model_layers_ae[li], input_dim=indim, activation=my_activation,name='enc_dns_'+str(li)))
        model.add(BatchNormalization(name='enc_bn_'+str(li)))
        model.add(Dropout(0.05,name='enc_dp_'+str(li)))
        indim = model_layers_ae[li]

    for layer in model.layers: # Encoder Network is used as is during Stage 2
        layer.trainable= False

    for li in range(model_layers_prd.size-1):
        model.add(Dense(model_layers_prd[li], input_dim=indim, activation=my_activation))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        indim = model_layers_prd[li]

    model.add(Dense(1, input_dim=indim, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model
