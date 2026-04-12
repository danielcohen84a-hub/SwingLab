import os
# Suppress TensorFlow logging spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, losses, metrics, optimizers

class SwingLabLSTM:

    def __init__(self, sequence_length=5, num_features=20, model_config=None):
        """
        Initializes the Neural Network architecture.
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        
        # Load hyperparams from config or use defaults
        self.config = model_config or {
            'learning_rate': 0.001,
            'dropout_rate': 0.2,
            'hidden_units': 64
        }
        
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.model = self._build_model()

    def _build_model(self):
        """
        Constructs the internal Keras Sequential Model.
        """
        model = models.Sequential()
        
        # 1. Input Layer: Defines the shape of our 3D tensor [batch, sequence, features]
        model.add(layers.InputLayer(shape=(self.sequence_length, self.num_features)))
        
        # 2. LSTM Engine: The main memory processor.
        model.add(layers.LSTM(units=self.config.get('hidden_units', 64), activation='tanh'))
        
        # 3. Pro Layer: Batch Normalization stabilizes the 35+ features during high-speed training.
        model.add(layers.BatchNormalization())
        
        # 4. Overfit Protection: 32% dropout as requested by the Auto-Tuner.
        model.add(layers.Dropout(self.config.get('dropout_rate', 0.2)))
        
        # 5. Pattern Translation: Concentrates the 96 LSTM outputs into a 32-unit decision layer.
        model.add(layers.Dense(units=32, activation='relu'))
        
        # 6. Final Output: Predicts 2 continuous numbers (swing return and scaled duration).
        model.add(layers.Dense(units=2, activation='linear'))
        
        # 6. Compilation: Huber loss protects against extreme market outliers (like Gamestop)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate), 
            loss=losses.Huber(), 
            metrics=[metrics.MeanAbsoluteError()]
        )
        
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, validation_split=0.2, callbacks=None):
        """
        Executes the training loop. If X_val is provided, it uses it for 
        chronological validation. Otherwise, it falls back to a random split.
        """
        from tensorflow.keras.callbacks import EarlyStopping
        
        # Stop training if the validation loss stops improving (prevents overfitting)
        training_callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        ]
        
        if callbacks:
            training_callbacks.extend(callbacks)
        
        val_data = (X_val, y_val) if (X_val is not None) else None
        
        print(f"Beginning Training on {len(X_train)} sequences...")
        if val_data:
            print(f"Using {len(X_val)} sequences for Chronological Validation (The 'Future')")

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            validation_split=validation_split if val_data is None else 0.0,
            callbacks=training_callbacks,
            shuffle=True, # Shuffling across 'Stories' is fine as long as the val-set is in the future
            verbose=1
        )
        return history

    def predict(self, X_live):
        """
        Uses the trained model to predict the next single swing slope.
        """
        return self.model.predict(X_live)

    def save_weights(self, filepath='models/swinglab_lstm2.weights.h5'):
        """Saves the trained neural network brain to the hard drive."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_weights(filepath)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath='models/swinglab_lstm2.weights.h5'):
        """Loads a frozen brain into the model (used for predict.py)"""
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
            print(f"Successfully loaded weights from {filepath}")
        else:
            print(f"WARNING: Could not find weights at {filepath}")

    def plot_training_history(self, history):
        """Visualizes how well the network learned over the epochs."""
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss (Huber)')
        plt.plot(history.history['val_loss'], label='Validation Loss (Huber)')
        plt.title('SwingLab LSTM Training Progress')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()