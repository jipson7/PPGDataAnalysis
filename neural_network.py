from keras.models import Sequential
from keras.layers import Dense


def get_model_generator(input_size):
    def create_model(hidden_layers=1, hidden_layer_size=input_size, optimizer='rmsprop', init='glorot_uniform'):
        # create model
        model = Sequential()
        model.add(Dense(hidden_layer_size, input_dim=input_size, kernel_initializer=init, activation='relu'))
        for i in range(hidden_layers - 1):
            model.add(Dense(hidden_layer_size, kernel_initializer=init, activation='relu'))
        model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    return create_model