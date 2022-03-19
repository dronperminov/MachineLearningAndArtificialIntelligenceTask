from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
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


def eval_model(model, features, test_labels, average='macro'):
    pred_labels = model.predict(features)

    precision = precision_score(test_labels, pred_labels, average=average, zero_division=0)
    recall = recall_score(test_labels, pred_labels, average=average, zero_division=0)
    f1 = f1_score(test_labels, pred_labels, average=average, zero_division=0)
    acc = model.score(features, test_labels)

    return precision, recall, f1, acc


def mean(values):
    return sum(values) / len(values)


def train(models, train_features, train_labels, test_features, test_labels):
    for name, model in models.items():
        model.fit(train_features, train_labels)

    print('|           Модель           | Recall | Precision | macro f1 | Accuracy |')
    print('|                        :-: |    :-: |       :-: |      :-: |      :-: |')

    for name, model in models.items():
        recall, precision, f1, acc = eval_model(model, test_features, test_labels)
        print(f'| {name:26} | {recall:6.4} | {precision:9.4} | {f1:8.4} | {acc:8.4} |')


def cross_val(models, features, labels, n_splits=10):
    for model_name, model in models.items():
        kfold = KFold(n_splits=n_splits)

        recalls, precisions, f1s, accuracies = [], [], [], []

        print(f'### {model_name}')
        print('| Разбиение | Recall | Precision | macro f1 | Accuracy |')
        print('|       :-: |    :-: |       :-: |      :-: |      :-: |')

        for fold, (train_index, test_index) in enumerate(kfold.split(features)):
            train_features, test_features = features.iloc[train_index], features.iloc[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]

            model = clone(model)
            model.fit(train_features, train_labels)

            name = f'{fold + 1} / {n_splits}'
            recall, precision, f1, accuracy = eval_model(model, test_features, test_labels)

            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            accuracies.append(accuracy)

            print(f'| {name:9} | {recall:6.4} | {precision:9.4} | {f1:8.4} | {accuracy:8.4} |')

        print(f'| в среднем | {mean(recalls):6.4} | {mean(precisions):9.4} | {mean(f1s):8.4} | {mean(accuracies):8.4} |\n')


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
        'Ridge': RidgeClassifier(),
        'MLP': MLPClassifier(random_state=random_state),
        'Decision tree': DecisionTreeClassifier(random_state=random_state),
        'Random forest': RandomForestClassifier(random_state=random_state),
        'Gradient boost': GradientBoostingClassifier(random_state=random_state),
        'XGBoost': XGBClassifier(use_label_encoder=False, verbosity=0, random_state=random_state),
        'CatBoost': CatBoostClassifier(verbose=False, random_state=random_state),
    }

    print('## Результаты кроссвалидации')
    cross_val(models, features, labels, n_splits=10)

    print('## Результаты обучения')
    train(models, train_features, train_labels, test_features, test_labels)


if __name__ == '__main__':
    main()
