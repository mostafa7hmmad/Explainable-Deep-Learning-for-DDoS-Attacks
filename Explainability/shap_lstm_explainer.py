import shap
import numpy as np
from tensorflow.keras.models import load_model
from explainability.utils import reshape_for_lstm

def explain_lstm_model(X_train_lstm, X_test_lstm, df, model_path="my_lstm_model.keras"):
    # Load model
    model = load_model(model_path)

    # Flatten LSTM input (samples, time_steps, features) → (samples, features)
    X_train_flat = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[2])
    X_test_flat = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[2])

    # Define prediction wrapper
    def model_predict(data_2d):
        return model.predict(reshape_for_lstm(data_2d))

    # Background for SHAP
    background = X_train_flat[np.random.choice(X_train_flat.shape[0], 100, replace=False)]

    # Create SHAP explainer
    explainer = shap.KernelExplainer(model_predict, background)

    # Subset of test samples
    X_test_subset = X_test_flat[:50]

    # Compute SHAP values (one set per class)
    shap_values = explainer.shap_values(X_test_subset)

    # Get feature names
    feature_columns = df.drop(['Target', 'ACK_Flag_Count', 'Init_Win_bytes_forward'], axis=1).columns.tolist()

    # Summary plots
    for i in range(len(shap_values)):
        print(f"Summary plot for class {i}")
        shap.summary_plot(shap_values[i], X_test_subset, feature_names=feature_columns)

    return shap_values
