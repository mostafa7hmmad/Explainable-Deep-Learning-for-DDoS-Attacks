from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_ann_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='sigmoid')
    ])
    return model
