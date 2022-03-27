from typing import List, Tuple
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
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


def print_average(average_results, sort_key=None, log_path: str = ''):
    max_len = max([len(name) for name in average_results])
    rows = [{'name': name, **results} for name, results in average_results.items()]

    if sort_key is not None:
        rows.sort(key=lambda x: x[sort_key])

    lines = [
        '### Усреднённые результаты кроссвалидации',
        f'| {"Модель":^{max_len}} | Recall | Precision | macro f1 | Accuracy |',
        f'| {":-:":>{max_len}} |    :-: |       :-: |      :-: |      :-: |'
    ]

    for row in rows:
        lines.append(f'| {row["name"]:{max_len}} | {row["Recall"]:6.4} | {row["Precision"]:9.4} | {row["f1"]:8.4} | {row["Accuracy"]:8.4} |')

    if log_path:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    else:
        print('\n'.join(lines))


def train(models, train_features, train_labels, test_features, test_labels):
    for name, model in models.items():
        model.fit(train_features, train_labels)

    print('|           Модель           | Recall | Precision | macro f1 | Accuracy |')
    print('|                        :-: |    :-: |       :-: |      :-: |      :-: |')

    for name, model in models.items():
        recall, precision, f1, acc = eval_model(model, test_features, test_labels)
        print(f'| {name:26} | {recall:6.4} | {precision:9.4} | {f1:8.4} | {acc:8.4} |')


def cross_val(models, features, labels, log_path: str = '', n_splits=10):
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

        if log_path:
            print_average(average_results, sort_key='f1', log_path=log_path)

        splits = ['в среднем'] + [f'{fold + 1} / {n_splits}' for fold in range(n_splits)]

        print(f'### {model_name}')
        print('| Разбиение  |' + ' | '.join(['%10s' % split for split in splits]) + ' |')
        print('|       :-:  |' + ' | '.join(['%10s' % ':-:' for _ in splits]) + ' |')
        for metric, values in metrics.items():
            print(f'| {metric:10} |' + ' | '.join(['%10.4f' % value for value in values]) + ' |')
        print()

    return average_results


def init_models(random_state):
    opt_xgboost_params = dict(
        booster="gbtree",
        tree_method="approx",
        learning_rate=0.1,
        max_depth=4,
        n_estimators=400,
        colsample_bynode=1,
        colsample_bytree=0.5,
    )

    opt_catboost_params = dict(
        learning_rate=0.25,
        max_depth=3,
        iterations=1000,
        l2_leaf_reg=3,
    )

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
        'XGBoost (optimal)': XGBClassifier(use_label_encoder=False, verbosity=0, random_state=random_state, **opt_xgboost_params),
        'CatBoost': CatBoostClassifier(verbose=False, random_state=random_state),
        'CatBoost (optimal)': CatBoostClassifier(verbose=False, random_state=random_state, **opt_catboost_params),
    }

    return models


def init_grid_xgboost_models(random_state):
    models = {
        'XGBoost default': XGBClassifier(use_label_encoder=False, verbosity=0, random_state=random_state),
    }

    lrs = [0.04, 0.08, 0.1, 0.25, 0.5]
    depths = [3, 4, 5]
    ns_estimators = [100, 250, 400, 600, 800]
    colsamples_bynode = [0.5, 0.8, 1]
    colsamples_bytree = [0.5, 0.8, 1]
    tree_methods = ['hist', 'approx', 'exact']

    for lr, depth, n_estimators, colsample_bynode, colsample_bytree, tree_method in product(lrs, depths, ns_estimators, colsamples_bynode, colsamples_bytree, tree_methods):
        models[f'XGBoost lr={lr} max_depth={depth} n_ests={n_estimators} cs_by_node={colsample_bynode} cs_by_tree={colsample_bytree} tree_method={tree_method}'] = XGBClassifier(
            use_label_encoder=False,
            verbosity=0,
            booster="gbtree",
            tree_method=tree_method,
            random_state=random_state,
            learning_rate=lr,
            max_depth=depth,
            n_estimators=n_estimators,
            colsample_bynode=colsample_bynode,
            colsample_bytree=colsample_bytree,
        )

    return models


def init_grid_catboost_models(random_state):
    models = {
        'CatBoost default': CatBoostClassifier(random_state=random_state, logging_level='Silent'),
    }

    lrs = [0.04, 0.08, 0.1, 0.25, 0.5]
    depths = [2, 3, 4, 5, 6]
    ns_iterations = [100, 250, 500, 800, 1000, 2000]
    l2_leaf_regs = [1, 3, 5, 10, 100]

    for lr, depth, iterations, l2_leaf_reg in product(lrs, depths, ns_iterations, l2_leaf_regs):
        models[f'CatBoost lr={lr} max_depth={depth} iterations={iterations} l2_leaf_reg={l2_leaf_reg}'] = CatBoostClassifier(
            random_state=random_state,
            learning_rate=lr,
            max_depth=depth,
            iterations=iterations,
            l2_leaf_reg=l2_leaf_reg,
            logging_level='Silent'
        )

    return models


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

    models = init_models(random_state)

    if find_boost_params:
        models.update(init_grid_xgboost_models(random_state))
        models.update(init_grid_catboost_models(random_state))

    print('## Результаты кроссвалидации')
    average_results = cross_val(models, features, labels, n_splits=10, log_path='results.txt')

    print(get_preprocess_name(preprocessing_mode))
    print_average(average_results, sort_key='f1')

    print('\n## Результаты обучения')
    train(models, train_features, train_labels, test_features, test_labels)


if __name__ == '__main__':
    main()
