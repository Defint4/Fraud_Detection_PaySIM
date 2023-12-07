import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class DetectFraud:
    def __init__(self):
        self.path_file = "file/data.csv"

    def readCsv(self, path_file):
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
        return pd.read_csv(path_file, dtype=dtypes)

    def feature_engineering(self, data):
        
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


    def train(self, path_file):
        
        data = self.readCsv(path_file)
        data = self.feature_engineering(data)

        X = data.drop(columns=['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest']) #train
        X = pd.get_dummies(X, columns=['type'], drop_first=True)  
        y = data['isFraud']   

        #sur/sous échantillonnage
        rus_ratio = {0: 80000, 1: y.sum()}  # Exemple: sous-échantillonner la classe 0 à 30 000 si elle est trop grande
        smote_ratio = {0: 80000, 1: 90000}  # Exemple: sur-échantillonner la classe 1 à 40 000

        rus = RandomUnderSampler(sampling_strategy=rus_ratio, random_state=42)
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)

        X_resampled, y_resampled = rus.fit_resample(X, y)
        X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)
        
        #séparation entrainement/test
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=42)

        model = XGBClassifier(
            random_state=42,  # Assure la reproductibilité des résultats
            use_label_encoder=False,  # Désactive l'encodeur de labels pour les versions récentes de XGBoost
            eval_metric='logloss',  # Utilise la perte logarithmique comme métrique d'évaluation
            booster='gbtree',  # Utilise des arbres de décision boostés comme base du modèle
            scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1],  # Équilibre les classes en donnant plus de poids à la classe minoritaire
            tree_method='hist'  # Utilise l'algorithme basé sur les histogrammes pour la construction d'arbres
        )

            #hyperparamètres
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        #choix des meilleurs hyperparamètres
        CV_model = RandomizedSearchCV(model, param_grid, cv=StratifiedKFold(5), n_jobs=-1, scoring='recall', random_state=42)
        
        #Entrainement du model
        CV_model.fit(X_train, y_train)

        # print("Meilleurs paramètres: ", CV_model.best_params_)
        best_model = CV_model.best_estimator_

        #enregistrement du model pour une utilisation futur
        joblib.dump(best_model, 'model/model_saved.pkl')


        predictions = best_model.predict(X_test)
        #écriture des résultat et de la précision du model en %
        print(classification_report(y_test, predictions))
        print(f"Précision : {accuracy_score(y_test, predictions)*100} %")



    def start(self):
        self.train(self.path_file)

if __name__ == "__main__":
    DetectFraud().start()
