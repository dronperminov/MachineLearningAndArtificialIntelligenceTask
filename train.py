from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import clone

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression, PassiveAggressiveClassifier, Perceptron
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


def preprocessing(features: pd.DataFrame, mode: str = '') -> pd.DataFrame:
    if mode == 'standardize':
        features = (features - features.mean()) / features.std()
    elif mode == 'normalize':
        features = (features - features.min()) / (features.max() - features.min())

    return features


def get_preprocess_name(preprocessing_mode: str) -> str:
    if preprocessing_mode == 'standardize':
        return 'стандартизация'

    if preprocessing_mode == 'normalize':
        return 'нормализация'

    return 'без изменений'


def eval_model(model, features, test_labels, average='macro'):
    pred_labels = model.predict(features)

    precision = precision_score(test_labels, pred_labels, average=average, zero_division=0)
    recall = recall_score(test_labels, pred_labels, average=average, zero_division=0)
    f1 = f1_score(test_labels, pred_labels, average=average, zero_division=0)
    acc = model.score(features, test_labels)

    return precision, recall, f1, acc


def mean(values):
    return sum(values) / len(values)


def print_average(average_results):
    print('### Усреднённые результаты кроссвалидации')
    print('|           Модель           | Recall | Precision | macro f1 | Accuracy |')
    print('|                        :-: |    :-: |       :-: |      :-: |      :-: |')

    for name, results in average_results.items():
        print(f'| {name:26} | {results["Recall"]:6.4} | {results["Precision"]:9.4} | {results["f1"]:8.4} | {results["Accuracy"]:8.4} |')


def get_optimal_parameters(name, model, parameters, features, labels):
    grid = GridSearchCV(estimator=clone(model), param_grid=parameters, cv=5, n_jobs=-1, verbose=1, scoring='f1_macro')
    grid.fit(features, labels)
    print(f'{name}:')
    print(f'    best score: {grid.best_score_}')
    print(f'    best parameters: {grid.best_params_}')
    return grid.best_params_


def train(models, train_features, train_labels, test_features, test_labels):
    for name, model in models.items():
        model.fit(train_features, train_labels)

    print('|           Модель           | Recall | Precision | macro f1 | Accuracy |')
    print('|                        :-: |    :-: |       :-: |      :-: |      :-: |')

    for name, model in models.items():
        recall, precision, f1, acc = eval_model(model, test_features, test_labels)
        print(f'| {name:26} | {recall:6.4} | {precision:9.4} | {f1:8.4} | {acc:8.4} |')


def cross_val(models, features, labels, n_splits=10):
    average_results = dict()

    for model_name, model in models.items():
        kfold = KFold(n_splits=n_splits)

        metrics = {
            'Recall': [0],
            'Precision': [0],
            'f1': [0],
            'Accuracy': [0]
        }

        for fold, (train_index, test_index) in enumerate(kfold.split(features)):
            train_features, test_features = features.iloc[train_index], features.iloc[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]

            model = clone(model)
            model.fit(train_features, train_labels)

            recall, precision, f1, accuracy = eval_model(model, test_features, test_labels)

            metrics['Recall'].append(recall)
            metrics['Precision'].append(precision)
            metrics['f1'].append(f1)
            metrics['Accuracy'].append(accuracy)

        average_results[model_name] = dict()
        for metric, values in metrics.items():
            avg = mean(values[1:])
            values[0] = avg
            average_results[model_name][metric] = avg

        splits = ['в среднем'] + [f'{fold + 1} / {n_splits}' for fold in range(n_splits)]

        print(f'### {model_name}')
        print('| Разбиение  |' + ' | '.join(['%10s' % split for split in splits]) + ' |')
        print('|       :-:  |' + ' | '.join(['%10s' % ':-:' for _ in splits]) + ' |')
        for metric, values in metrics.items():
            print(f'| {metric:10} |' + ' | '.join(['%10.4f' % value for value in values]) + ' |')
        print()

    return average_results


def main():
    random_state = 42
    preprocessing_mode = ''  # normalize, standardize
    find_boost_params = False

    dataset_path = 'tz_dataset.csv'
    dataset = read_dataset(dataset_path, unused_columns=['text', 'uid', 'group', 'Unnamed: 0'])

    dataset['is_in_toc'] = dataset['is_in_toc'].fillna(0)

    features, labels = get_data(dataset)
    features = preprocessing(features, preprocessing_mode)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=random_state)

    boost_grid_parameters = {
        'random_state': [random_state],
        'learning_rate': [0.01, 0.025, 0.04, 0.08, 0.1, 0.25, 0.5, 0.8],
        'n_estimators': [100, 250, 500, 600, 800, 1000],
        'max_depth': [2, 3, 4, 5, 6]
    }

    models = {
        'Logistic regression': LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=random_state),
        'Nearest centroid': NearestCentroid(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Passive aggressive': PassiveAggressiveClassifier(random_state=random_state),
        'SVM': SVC(random_state=random_state),
        'Ridge': RidgeClassifier(random_state=random_state),
        'Perceptron': Perceptron(random_state=random_state),
        'Multi layer perceptron': MLPClassifier(random_state=random_state),
        'Decision tree': DecisionTreeClassifier(random_state=random_state),
        'Random forest': RandomForestClassifier(random_state=random_state),
        'Gradient boost': GradientBoostingClassifier(random_state=random_state),
        'XGBoost': XGBClassifier(use_label_encoder=False, verbosity=0, random_state=random_state),
        'CatBoost': CatBoostClassifier(verbose=False, random_state=random_state),
    }

    if find_boost_params:
        xg_boost_params = get_optimal_parameters('XGBoost', models['XGBoost'], boost_grid_parameters, features, labels)
        cat_boost_params = get_optimal_parameters('CatBoost', models['CatBoost'], boost_grid_parameters, features, labels)

        models['XGBoost (optimal)'] = XGBClassifier(use_label_encoder=False, verbosity=0, **xg_boost_params)
        models['CatBoost (optimal)'] = CatBoostClassifier(verbose=False, **cat_boost_params)

    print('## Результаты кроссвалидации')
    average_results = cross_val(models, features, labels, n_splits=10)

    print(get_preprocess_name(preprocessing_mode))
    print_average(average_results)

    print('\n## Результаты обучения')
    train(models, train_features, train_labels, test_features, test_labels)


if __name__ == '__main__':
    main()
