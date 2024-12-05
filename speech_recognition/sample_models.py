from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, AveragePooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation, return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride, dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    simp_rnn1 = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn1')(input_data)
    bn_rnn1 = BatchNormalization(name='bn_simp_rnn1')(simp_rnn1)
    simp_rnn2 = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn2')(bn_rnn1)
    bn_rnn2 = BatchNormalization(name='bn_simp_rnn2')(simp_rnn2)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn2)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(output_dim, return_sequences=True))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def experimental_model(input_dim, filters, kernel_size, stride, border_mode, units, output_dim=29):

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=stride, 
                     padding=border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    
    bn_conv_1d = BatchNormalization(name='bn_conv_1d')(conv_1d)

    drop1 = Dropout(0.2)(bn_conv_1d)
    
    bidir_rnn1 = Bidirectional(GRU(output_dim, return_sequences=True), name="bidir_rnn1")(drop1)
    bidir_rnn2 = Bidirectional(GRU(output_dim, return_sequences=True), name="bidir_rnn2")(bidir_rnn1)

    bn_rnn = BatchNormalization(name='bn_rnn')(bidir_rnn2)

    drop2 = Dropout(0.2)(bn_rnn)
    
    time_dense = TimeDistributed(Dense(output_dim))(drop2)

    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, border_mode, stride)
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, stride, border_mode, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    conv_1 = Conv1D(filters, kernel_size, strides=stride, padding=border_mode, activation='relu', name='conv_1')(input_data)
    
    drop_1 = Dropout(0.2, name='drop_1')(conv_1)
    
    bn_1 = BatchNormalization(name='bn_1')(drop_1)

    bi_gru_1 = Bidirectional(GRU(output_dim, return_sequences=True), name="bi_gru_1")(bn_1)
    
    bn_2 = BatchNormalization(name='bn_2')(bi_gru_1)
    
    bi_gru_2 = Bidirectional(GRU(output_dim, return_sequences=True), name="bi_gru_2")(bn_2)

    bn_3 = BatchNormalization(name='bn_3')(bi_gru_2)
    
    bi_gru_3 = Bidirectional(GRU(output_dim, return_sequences=True), name="bi_gru_3")(bn_3)

    bn_4 = BatchNormalization(name='bn_4')(bi_gru_3)

    time_dense_1 = TimeDistributed(Dense(output_dim), name="time_dense_1")(bn_4)
    
    time_dense_2 = TimeDistributed(Dense(output_dim), name="time_dense_2")(time_dense_1)
    
    dense_1 = Dense(output_dim, name="dense_1")(time_dense_2)
    
    dense_2 = Dense(output_dim, name="dense_2")(dense_1)
    
    y_pred = Activation('softmax', name='softmax')(dense_2)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, border_mode, stride)
    print(model.summary())
    return model

def old_final_model(input_dim, filters, kernel_size, stride, border_mode, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=stride, 
                     padding=border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    
    bn_conv_1d = BatchNormalization(name='bn_conv_1d')(conv_1d)

    drop1 = Dropout(0.25)(bn_conv_1d)
    
    bidir_rnn1 = Bidirectional(GRU(output_dim, return_sequences=True), name="bidir_rnn1")(drop1)
    bidir_rnn2 = Bidirectional(GRU(output_dim, return_sequences=True), name="bidir_rnn2")(bidir_rnn1)

    bn_rnn = BatchNormalization(name='bn_rnn')(bidir_rnn2)

    drop2 = Dropout(0.25)(bn_rnn)

    time_dense1 = TimeDistributed(Dense(2*output_dim))(drop2)
    time_dense2 = TimeDistributed(Dense(output_dim))(time_dense1)
    
    drop3 = Dropout(0.25)(time_dense2)
    
    y_pred = Activation('softmax', name='softmax')(drop3)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, border_mode, stride)
    print(model.summary())
    return model