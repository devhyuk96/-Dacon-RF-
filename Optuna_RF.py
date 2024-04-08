import pandas as pd
submit = pd.read_csv('./sample_submission.csv')
train = pd.read_csv("./train.csv")

# train
pip install optuna

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
import optuna

# person_id 컬럼 제거
X_train = train.drop(['person_id', 'login'], axis=1)
y_train = train['login']

def objective(trial):
    param = {
        # 기본값: 10
        # 범위: 10 ~ 1000 사이의 양의 정수. 일반적으로 값이 클수록 모델 성능이 좋아지지만, 계산 비용과 시간도 증가합니다.
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),

        # 기본값: 'gini'
        # 옵션: 'gini', 'entropy'. 'gini'는 진니 불순도를, 'entropy'는 정보 이득을 기준으로 합니다.
        'criterion': 'gini',

        # 기본값: None
        # 범위: None 또는 양의 정수. None으로 설정하면 노드가 모든 리프가 순수해질 때까지 확장됩니다. 양의 정수를 설정하면 트리의 최대 깊이를 제한합니다.
        'max_depth' : trial.suggest_int('max_depth', 1, 100),

        # 기본값: 2
        # 범위: 2 이상의 정수 또는 0과 1 사이의 실수 (비율을 나타냄, (0, 1] ). 내부 노드를 분할하기 위해 필요한 최소 샘플 수를 지정합니다.
        # 'min_samples_split': trial.suggest_float('min_samples_split', 0.0, 1.0),  # 1보다 큰 값으로 설정

        # 기본값: 1
        # 범위: 1 이상의 정수 또는 0과 0.5 사이의 실수 (비율을 나타냄, (0, 0.5] ). 리프 노드가 가져야 하는 최소 샘플 수를 지정합니다.
        # 'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.0, 0.5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),  # 1보다 큰 값으로 설정
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),

        # 기본값: 0.0
        # 범위: 0.0에서 0.5 사이의 실수. 리프 노드에 있어야 하는 샘플의 최소 가중치 비율을 지정합니다.
        'min_weight_fraction_leaf':  trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),

        # 기본값: 'auto'
        # 옵션: 'auto', 'sqrt', 'log2', None 또는 양의 정수/실수.
        # 최적의 분할을 찾기 위해 고려할 특성의 수 또는 비율을 지정합니다.
        # 'auto'는 모든 특성을 사용함을 의미하며,
        # 'sqrt'와 'log2'는 각각 특성의 제곱근과 로그2를 사용합니다.
        # None은 'auto'와 동일하게 모든 특성을 의미합니다.
        # 'max_features':'log2',
        'max_features': 'sqrt',

        # 기본값: None
        # 범위: None 또는 양의 정수. 리프 노드의 최대 수를 제한합니다. None은 무제한을 의미합니다.
        'max_leaf_nodes':  trial.suggest_int('max_leaf_nodes', 1, 100),

        # 기본값: 0.0
        # 범위: 0.0 이상의 실수. 노드를 분할할 때 감소해야 하는 불순도의 최소량을 지정합니다.
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 1.0),

        # 기본값: True
        # 옵션: True, False. True는 부트스트랩 샘플을 사용하여 개별 트리를 학습시킵니다.
        # False는 전체 데이터셋을 사용하여 각 트리를 학습시킵니다.
        'bootstrap': True,
        'n_jobs': -1,
        'random_state': 42
    }

    model = RandomForestClassifier(**param)

    cv = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=KFold(n_splits=3)).mean()

    return cv

%%time
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10000)

# optuna가 시도했던 모든 실험 관련 데이터
study.trials_dataframe()

# optuna best parameter 
print('Best trial: score {}, \nparams {}'.format(study.best_trial.value,study.best_trial.params))

# Hyperparameter Importances를 통해서 parameter를 고정시켜라.
# 그리고 나머지 것들을 진행시켜라.
optuna.visualization.plot_param_importances(study)

optuna.visualization.plot_optimization_history(study)

best_params = study.best_params

# submission
# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('./baseline_submit.csv', index=False)
