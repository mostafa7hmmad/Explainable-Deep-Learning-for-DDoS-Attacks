from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN
from tensorflow.keras.optimizers import Adam

# ===================== Build ANN Model =====================
def build_ann_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# ===================== Build LSTM Model =====================
def build_lstm_model(rnn_shape, num_classes):
    model = Sequential([
        LSTM(128, activation='tanh', input_shape=rnn_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(64, activation='tanh'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# ===================== Build RNN Model =====================
def build_rnn_model(rnn_shape, num_classes):
    model = Sequential([
        SimpleRNN(64, activation='tanh', input_shape=rnn_shape, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(32, activation='tanh'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# ===================== Train ANN =====================
def train_ann_model(X_train, y_train_cat, X_val, y_val_cat,
                    input_shape, num_classes, optimizer,
                    epochs=50, batch_size=32):
    model = build_ann_model(input_shape, num_classes)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size
    )
    return model, history

# ===================== Train LSTM =====================
def train_lstm_model(X_train_rnn, y_train_cat, X_val_rnn, y_val_cat,
                     rnn_shape, num_classes,
                     epochs=50, batch_size=32):
    model = build_lstm_model(rnn_shape, num_classes)
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(
        X_train_rnn, y_train_cat,
        validation_data=(X_val_rnn, y_val_cat),
        epochs=epochs,
        batch_size=batch_size
    )
    return model, history

# ===================== Train RNN =====================
def train_rnn_model(X_train_rnn, y_train_cat, X_val_rnn, y_val_cat,
                    rnn_shape, num_classes,
                    epochs=50, batch_size=32):
    model = build_rnn_model(rnn_shape, num_classes)
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(
        X_train_rnn, y_train_cat,
        validation_data=(X_val_rnn, y_val_cat),
        epochs=epochs,
        batch_size=batch_size
    )
    return model, history
