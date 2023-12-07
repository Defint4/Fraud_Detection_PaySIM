import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
import joblib

class DetectFraud:
    def __init__(self, path_file):
        self.path_file = path_file

    def load_and_preprocess_data(self):
        data = self._load_data()
        data = self._feature_engineering(data)
        return self._prepare_data_for_training(data)

    def _load_data(self):
        dtypes = {
            'step': 'int16',
            'amount': 'float32',
            'oldbalanceOrg': 'float32',
            'newbalanceOrig': 'float32',
            'oldbalanceDest': 'float32',
            'newbalanceDest': 'float32',
            'isFraud': 'int8',
            'isFlaggedFraud': 'int8'
        }
        return pd.read_csv(self.path_file, dtype=dtypes)

    def _feature_engineering(self, data):
        
        # différence de solde dans les comptes avant et après la transaction
        data['delta_balance_orig'] = data['oldbalanceOrg'] - data['newbalanceOrig']
        data['delta_balance_dest'] = data['oldbalanceDest'] - data['newbalanceDest']

        #Indicateurs de solde avant et après la transaction (s'il est nul ou pas)
        data['orig_bal_zero'] = (data['oldbalanceOrg'] == 0).astype(int)      #envoi des int et non des bool
        data['new_orig_bal_zero'] = (data['newbalanceOrig'] == 0).astype(int)
        
        data['dest_bal_zero'] = (data['oldbalanceDest'] == 0).astype(int)
        data['new_dest_bal_zero'] = (data['newbalanceDest'] == 0).astype(int)


        #Ratios de montant par rapport aux balances initiales
        data['amount_to_oldbalanceOrg_ratio'] = data['amount'] / (data['oldbalanceOrg'] + 1)  # +1 pour éviter la division par zéro
        data['amount_to_oldbalanceDest_ratio'] = data['amount'] / (data['oldbalanceDest'] + 1)

        return data

    def _prepare_data_for_training(self, data):
        X = data.drop(columns=['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest'])
        X = pd.get_dummies(X, columns=['type'], drop_first=True)
        y = data['isFraud']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.20, random_state=42)

    def build_and_compile_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_and_evaluate_model(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train, epochs=1, batch_size=64, validation_split=0.2)
        
        predictions = model.predict(X_test)
        predictions = (predictions > 0.5).astype(int)

        print(classification_report(y_test, predictions))
        print(f"Précision : {accuracy_score(y_test, predictions)*100} %")

        model.save('model/neural_network_model_optimized.h5')

    def start(self):
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        model = self.build_and_compile_model(X_train.shape[1])
        self.train_and_evaluate_model(model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    fraud_detector = DetectFraud("file/data.csv")
    fraud_detector.start()
