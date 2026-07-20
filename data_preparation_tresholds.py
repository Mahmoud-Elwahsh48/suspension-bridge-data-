import numpy as np
import pandas as pd
import logging
import joblib
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns  # <-- Add this import


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# Aliases
layers = tf.keras.layers
Model = tf.keras.Model
Input = tf.keras.Input

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

THRESHOLD_FILE = "thresholds.json"

# ─── Corrosion calculation ─────────────────────────────────────────────────────
def calculate_corrosion(t, T, RH, TOW, Precip, pH, Cl, material_type='steel'):
    if material_type == 'steel':
        A_coeff = np.random.normal(85, 10)
        n = np.random.normal(0.35, 0.05)
    else:
        A_coeff = np.random.normal(65, 8)
        n = np.random.normal(0.28, 0.03)

    env_factor = (
        0.15 * T
        + 0.12 * RH
        + 0.10 * TOW
        + 0.05 * Precip / 1000.0
        + 0.18 * (7 - pH)
        + 0.30 * np.log(Cl + 1)
    )

    A = A_coeff * (1 + env_factor)
    C = A * (t ** n)
    return C / 1000, A / 1000  # scale to mm

# ─── File Reading ───────────────────────────────────────────────────────────────
def read_file():
    file_path = 'Bridge data.csv'
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None

# ─── Save & Load Thresholds ─────────────────────────────────────────────────────
def save_thresholds(thresholds):
    with open(THRESHOLD_FILE, "w") as f:
        json.dump(thresholds, f, indent=4)
    logging.info(f"💾 Thresholds saved to {THRESHOLD_FILE}")

def load_thresholds():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r") as f:
            return json.load(f)
    return None

# ─── Data Collection and Dynamic Thresholds ─────────────────────────────────────
def collect_data():
    data = read_file()
    if data is None:
        return

    # Estimate cable exposure time as 80% of bridge age
    t = data['BridgeAge'] * 0.8

    # Apply corrosion function
    results = data.apply(lambda row: calculate_corrosion(
        t=row['BridgeAge'] * 0.8,
        T=row['Average_Temperature'],
        RH=row['Average_Humidity'],
        TOW=row['Mean_TOW'],
        Precip=row['TotalPrecipitation'],
        pH=row['Mean_PH'],
        Cl=row['Mean_Cl-'],
        material_type='steel'
    ), axis=1)

    data['CorrosionDepth'], data['AnnualRate'] = zip(*results)

    # Derived features
    data['Wind-Induced Vibrations'] = data['Mean_Wind_Speed'] * np.sin(np.radians(data['Wind_Direction']))
    data['Traffic Load'] = data['AverageDailyTraffic'] * data['Structure_Length']

    # Structural Health Index
    data['Structural Health'] = 100 - (
        (data['BridgeAge'] * 0.1) +
        (data['Wind-Induced Vibrations'] * 0.2) +
        (data['AnnualRate'] * 0.3) +
        ((data['Traffic Load'] / 1e6) * 0.4)
    )

    # ✅ Dynamic thresholds (or load if exist)
    thresholds = load_thresholds()
    if thresholds is None:
        thresholds = {
            'Wind_Induced_Vibrations Risk': np.percentile(data['Wind-Induced Vibrations'], [25, 50, 75]).tolist(),
            'Corrosion Risk': np.percentile(data['AnnualRate'], [25, 50, 75]).tolist(),
            'Traffic_Load Risk': np.percentile(data['Traffic Load'], [25, 50, 75]).tolist(),
            'Structural_Health Risk': np.percentile(data['Structural Health'], [25, 50, 75]).tolist()
        }
        save_thresholds(thresholds)

    logging.info(f"📊 Dynamic Thresholds: {thresholds}")

    # Apply risk categories
    data['Vibration Risk'] = pd.cut(
        data['Wind-Induced Vibrations'],
        bins=[-np.inf] + thresholds['Wind_Induced_Vibrations Risk'] + [np.inf],
        labels=['Low', 'Moderate', 'High', 'Critical']
    )

    data['Corrosion Risk'] = pd.cut(
        data['AnnualRate'],
        bins=[-np.inf] + thresholds['Corrosion Risk'] + [np.inf],
        labels=['Low', 'Moderate', 'High', 'Critical']
    )

    max_design = data['Traffic Load'].max()
    data['Traffic Load Percentage'] = (data['Traffic Load'] / max_design) * 100
    data['Traffic Load Risk'] = pd.cut(
        data['Traffic Load'],
        bins=[-np.inf] + thresholds['Traffic_Load Risk'] + [np.inf],
        labels=['Low', 'Moderate', 'High', 'Critical']
    )

    data['Structural Health Risk'] = pd.cut(
        data['Structural Health'],
        bins=[-np.inf] + thresholds['Structural_Health Risk'] + [np.inf],
        labels=['Critical', 'Poor', 'Fair', 'Good']
    )

    # Save
    output_path = 'bridge_data1.csv'
    data.to_csv(output_path, index=False)
    logging.info(f"✅ File saved: {output_path}")
    return data

# ─── Classification Training ───────────────────────────────────────────────────
# ─── Classification Training ───────────────────────────────────────────────────
def training_classify_data():
    data = collect_data()

    # Encode categorical risks
    label_encoders = {}
    for column in ['Vibration Risk', 'Corrosion Risk', 'Traffic Load Risk', 'Structural Health Risk']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    joblib.dump(label_encoders, 'label_encoders.pkl')

    # Drop risk labels from features
    X = data.drop(columns=['Vibration Risk', 'Corrosion Risk', 'Traffic Load Risk', 'Structural Health Risk'])
    X = X.select_dtypes(include=[np.number])

    y_class = data[['Vibration Risk', 'Corrosion Risk', 'Traffic Load Risk', 'Structural Health Risk']]

    # Train-test split
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')

    # One-hot encode targets
    y_train_class_oh = [tf.keras.utils.to_categorical(y_train_class.iloc[:, i], num_classes=4) for i in range(4)]
    y_test_class_oh = [tf.keras.utils.to_categorical(y_test_class.iloc[:, i], num_classes=4) for i in range(4)]

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    # Model
    def build_classification_model(input_dim):
        input_layer = Input(shape=(input_dim,))
        x = layers.Dense(256, activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        outputs = [
            layers.Dense(4, activation='softmax', name='Vibration_Risk')(x),
            layers.Dense(4, activation='softmax', name='Corrosion_Risk')(x),
            layers.Dense(4, activation='softmax', name='Traffic_Load_Risk')(x),
            layers.Dense(4, activation='softmax', name='Structural_Health_Risk')(x)
        ]

        model = Model(inputs=input_layer, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'Vibration_Risk': 'categorical_crossentropy',
                'Corrosion_Risk': 'categorical_crossentropy',
                'Traffic_Load_Risk': 'categorical_crossentropy',
                'Structural_Health_Risk': 'categorical_crossentropy'
            },
            metrics={
                'Vibration_Risk': 'accuracy',
                'Corrosion_Risk': 'accuracy',
                'Traffic_Load_Risk': 'accuracy',
                'Structural_Health_Risk': 'accuracy'
            }
        )

        return model

    classification_model = build_classification_model(X_train.shape[1])

    logging.info("Training classification model...")
    history = classification_model.fit(
        X_train, y_train_class_oh,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    classification_model.save('classification_model_dynamic.h5')

    # ─── Evaluation Metrics ───────────────────────────────────────────────────
    logging.info("Evaluating model performance...")
    y_pred_class = classification_model.predict(X_test)
    
    # Class names for each risk type
    class_names = {
        'Vibration Risk': ['Low', 'Moderate', 'High', 'Critical'],
        'Corrosion Risk': ['Low', 'Moderate', 'High', 'Critical'],
        'Traffic Load Risk': ['Low', 'Moderate', 'High', 'Critical'],
        'Structural Health Risk': ['Critical', 'Poor', 'Fair', 'Good']
    }
    
    # Create evaluation directory if it doesn't exist
    os.makedirs('evaluation_metrics', exist_ok=True)
    
    # Evaluate each risk type separately
    for i, risk_type in enumerate(['Vibration Risk', 'Corrosion Risk', 'Traffic Load Risk', 'Structural Health Risk']):
        # Get true and predicted labels
        y_true = y_test_class.iloc[:, i]
        y_pred = np.argmax(y_pred_class[i], axis=1)
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Save metrics to file
        with open(f'evaluation_metrics/{risk_type}_metrics.txt', 'w') as f:
            f.write(f"Evaluation Metrics for {risk_type}\n")
            f.write("="*40 + "\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall: {rec:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write("Class Labels:\n")
            for idx, label in enumerate(class_names[risk_type]):
                f.write(f"{idx}: {label}\n")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names[risk_type], 
                    yticklabels=class_names[risk_type])
        plt.title(f'Confusion Matrix for {risk_type}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'evaluation_metrics/{risk_type}_confusion_matrix.png')
        plt.close()
        
        # Print metrics to console
        logging.info(f"\n{risk_type} Evaluation:")
        logging.info(f"Accuracy: {acc:.4f}")
        logging.info(f"Precision: {prec:.4f}")
        logging.info(f"Recall: {rec:.4f}")
        logging.info("Confusion Matrix:")
        logging.info(cm)
        logging.info("Class Labels:")
        for idx, label in enumerate(class_names[risk_type]):
            logging.info(f"{idx}: {label}")
        logging.info("-"*50)
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('evaluation_metrics/training_history.png')
    plt.close()
    
    logging.info("All evaluation metrics and plots saved in 'evaluation_metrics' directory")

# ─── Regression Training ───────────────────────────────────────────────────────
def training_regression_data():
    data = collect_data()

    # Target: degradation factor
    degradation_factors = (
        (data['BridgeAge'] * 0.1) +
        (data['Wind-Induced Vibrations'] * 0.2) +
        (data['AnnualRate'] * 0.3) +
        ((data['Traffic Load'] / 1e6) * 0.4)
    )

    X = data.drop(columns=['Structural Health', 'Vibration Risk', 'Corrosion Risk', 'Traffic Load Risk', 'Structural Health Risk'])
    X = X.select_dtypes(include=[np.number])

    y = degradation_factors

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'regression_scaler.pkl')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    def build_regression_model(input_dim):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mean_squared_error', metrics=['mae'])
        return model

    regression_model = build_regression_model(X_train_scaled.shape[1])

    logging.info("Training regression model...")
    history = regression_model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Regression Training History')
    plt.show()

    regression_model.save('regression_model_dynamic.h5')

    y_pred_degradation = regression_model.predict(X_test_scaled).flatten()
    y_pred_health = 100 - y_pred_degradation
    y_true_health = 100 - y_test

    mae = mean_absolute_error(y_true_health, y_pred_health)
    logging.info(f"Health Score MAE: {mae:.2f}")

    sample_results = pd.DataFrame({
        'Actual Health': y_true_health[:5],
        'Predicted Health': y_pred_health[:5],
        'Difference': y_pred_health[:5] - y_true_health[:5]
    })
    print(sample_results)

# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    collect_data()
    training_classify_data()
    training_regression_data()
