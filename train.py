from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def read_dataset(dataset_path: str, unused_columns) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)
    dataset = dataset.drop(columns=unused_columns)
    return dataset


def get_data(dataset: pd.DataFrame, label_key: str = 'label') -> Tuple[pd.DataFrame, List[str]]:
    labels = [label for label in dataset[label_key]]
    labels = LabelEncoder().fit_transform(labels)
    features = dataset.drop(columns=[label_key])
    return features, labels


def eval_model(name, model, features, test_labels, average='macro'):
    pred_labels = model.predict(features)

    precision = precision_score(test_labels, pred_labels, average=average)
    recall = recall_score(test_labels, pred_labels, average=average)
    f1 = f1_score(test_labels, pred_labels, average=average)
    acc = model.score(features, test_labels)

    print(f'| {name:15} | {recall:6.4} | {precision:9.4} | {f1:8.4} | {acc:8.4} |')


def main():
    random_state = 42
    dataset_path = 'tz_dataset.csv'
    dataset = read_dataset(dataset_path, unused_columns=['text', 'uid', 'group', 'Unnamed: 0'])

    dataset['is_in_toc'] = dataset['is_in_toc'].fillna(0)

    features, labels = get_data(dataset)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=random_state)

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(random_state=random_state),
        'MLP': MLPClassifier(random_state=random_state),
        'Decision tree': DecisionTreeClassifier(random_state=random_state),
        'Random forest': RandomForestClassifier(random_state=random_state),
        'XGBoost': XGBClassifier(use_label_encoder=False, verbosity=0, random_state=random_state),
        'CatBoost': CatBoostClassifier(verbose=False, random_state=random_state),
    }

    print('|      Model      | Recall | Precision | macro f1 | accuracy |')
    print('+-----------------+--------+-----------+----------+----------+')
    for name, model in models.items():
        model.fit(train_features, train_labels)
        eval_model(name, model, test_features, test_labels)


if __name__ == '__main__':
    main()
