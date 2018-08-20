from keras.models import Sequential
from keras.layers import Dense


def get_model_generator(input_size):
    def create_model(optimizer='adam', init='normal'):
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=input_size, kernel_initializer=init, activation='relu'))
        model.add(Dense(4, kernel_initializer=init, activation='relu'))
        model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    return create_model
