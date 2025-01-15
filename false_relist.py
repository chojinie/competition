from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 데이터 불러오기
train_data = pd.read_csv('train.csv')  # 학습 데이터
test_data = pd.read_csv('test.csv')    # 테스트 데이터

# 범주형 데이터 인코딩 (학습 데이터)
one_hot_cols = ['매물확인방식', '방향', '주차가능여부', '중개사무소', '제공플랫폼']  # One-Hot Encoding 대상 컬럼

# One-Hot Encoding
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
train_encoded = one_hot_encoder.fit_transform(train_data[one_hot_cols]).toarray()
train_encoded_df = pd.DataFrame(train_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols), index=train_data.index)
train_data = pd.concat([train_data.drop(columns=one_hot_cols), train_encoded_df], axis=1)

test_encoded = one_hot_encoder.transform(test_data[one_hot_cols]).toarray()
test_encoded_df = pd.DataFrame(test_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols), index=test_data.index)
test_data = pd.concat([test_data.drop(columns=one_hot_cols), test_encoded_df], axis=1)

# 날짜 데이터 처리 (학습 데이터)
train_data['게재일'] = pd.to_datetime(train_data['게재일'])
test_data['게재일'] = pd.to_datetime(test_data['게재일'])

# 게재일을 기준으로 KMeans 클러스터링 수행
train_dates = (train_data['게재일'] - train_data['게재일'].min()).dt.days.values.reshape(-1, 1)
test_dates = (test_data['게재일'] - train_data['게재일'].min()).dt.days.values.reshape(-1, 1)

kmeans_date = KMeans(n_clusters=5, random_state=42)  # 클러스터 수 5로 설정
train_data['게재일_클러스터'] = kmeans_date.fit_predict(train_dates)
test_data['게재일_클러스터'] = kmeans_date.predict(test_dates)

# 게재일 관련 피처 추가
train_data['게재연도'] = train_data['게재일'].dt.year
train_data['게재월'] = train_data['게재일'].dt.month
train_data['게재일자'] = train_data['게재일'].dt.day
train_data['게재요일'] = train_data['게재일'].dt.weekday
train_data.drop(columns=['게재일'], inplace=True)

test_data['게재연도'] = test_data['게재일'].dt.year
test_data['게재월'] = test_data['게재일'].dt.month
test_data['게재일자'] = test_data['게재일'].dt.day
test_data['게재요일'] = test_data['게재일'].dt.weekday
test_data.drop(columns=['게재일'], inplace=True)

# 피처 엔지니어링 (보증금, 월세, 관리비를 활용한 새로운 특성 생성)
train_data['월세_비율'] = train_data['월세'] / train_data['보증금']
train_data['관리비_비율'] = train_data['관리비'] / train_data['보증금']
train_data['층_비율'] = train_data['해당층'] / train_data['총층']

train_data['관리비_구간'] = '0'  # 관리비가 0인 경우, 범주 '0'으로 설정
train_data.loc[train_data['관리비'] > 0, '관리비_구간'] = pd.cut(
    train_data.loc[train_data['관리비'] > 0, '관리비'],
    bins=[0, 5, 10, 15, 20, 30, 100],  # 구간 경계값 지정
    labels=['1', '2', '3', '4', '5', '6'],  # 구간 레이블
    right=False
)
train_data['보증금_로그'] = np.log1p(train_data['보증금'])

test_data['월세_비율'] = test_data['월세'] / test_data['보증금']
test_data['관리비_비율'] = test_data['관리비'] / test_data['보증금']
test_data['층_비율'] = test_data['해당층'] / test_data['총층']

test_data['관리비_구간'] = '0'
test_data.loc[test_data['관리비'] > 0, '관리비_구간'] = pd.cut(
    test_data.loc[test_data['관리비'] > 0, '관리비'],
    bins=[0, 5, 10, 15, 20, 30, 100],
    labels=['1', '2', '3', '4', '5', '6'],
    right=False
)
test_data['보증금_로그'] = np.log1p(test_data['보증금'])

# 수치형 데이터만 선택하여 클러스터링 기반 특성 생성
numeric_columns = ['보증금', '월세', '전용면적', '해당층', '총층', '방수', '욕실수', '총주차대수']
numeric_df = train_data[numeric_columns].fillna(0)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# k-means 클러스터링 적용 (최적 k=7 사용)
kmeans = KMeans(n_clusters=7, random_state=42)
train_data['cluster_id'] = kmeans.fit_predict(scaled_data)

# 테스트 데이터에도 동일한 클러스터링 적용
test_numeric_df = test_data[numeric_columns].fillna(0)
test_scaled_data = scaler.transform(test_numeric_df)
test_data['cluster_id'] = kmeans.predict(test_scaled_data)

# 3단계: 클러스터별 비율 피처 추가
cluster_stats = train_data.groupby('cluster_id')[numeric_columns].mean()

for col in numeric_columns:
    train_data[f'{col}_cluster_ratio'] = train_data.apply(lambda x: x[col] / cluster_stats.loc[x['cluster_id'], col] if cluster_stats.loc[x['cluster_id'], col] != 0 else 0, axis=1)
    test_data[f'{col}_cluster_ratio'] = test_data.apply(lambda x: x[col] / cluster_stats.loc[x['cluster_id'], col] if cluster_stats.loc[x['cluster_id'], col] != 0 else 0, axis=1)

# 특성과 레이블 분리
X = train_data.drop(columns=['ID', '허위매물여부'])
y = train_data['허위매물여부']

# 테스트 데이터를 위한 전처리
X_test = test_data.drop(columns=['ID'])

# KNNImputer를 적용할 컬럼 지정 (결측치가 있는 컬럼만)
columns_fill_knn = ['전용면적', '해당층', '총층', '방수', '욕실수', '총주차대수']

# 수치형 컬럼 지정 (스케일링 대상)
numerical_cols = ['보증금', '월세', '월세_비율', '관리비_비율', '보증금_로그', '층_비율'] + [f'{col}_cluster_ratio' for col in numeric_columns]

# 날짜 클러스터 및 추가 피처 포함한 수치형/범주형 컬럼 정의
numerical_cols += ['게재일_클러스터']
categorical_cols = ['관리비_구간', '게재연도', '게재월', '게재일자', '게재요일']

# 전처리 파이프라인 설정 (수정)
preprocessor = ColumnTransformer(
    transformers=[
        ('knn_imputer', KNNImputer(n_neighbors=7), columns_fill_knn),  # 결측치 보완
        ('scaler', StandardScaler(), numerical_cols),                  # 수치형 데이터 스케일링
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # 범주형 데이터 원-핫 인코딩
    ],
    remainder='passthrough'
)

# Bayesian Optimization을 위한 목적 함수 정의
def xgb_cv(n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_weight):
    # 함수 내부에서 새로운 모델 생성
    model = XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',
        device='cuda:0'
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for train_index, val_index in kf.split(X):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 전처리 적용
        X_tr_preprocessed = preprocessor.fit_transform(X_tr)
        X_val_preprocessed = preprocessor.transform(X_val)
        
        # 모델 학습
        model.fit(X_tr_preprocessed, y_tr)
        
        # 검증 데이터로 예측
        y_val_pred = model.predict(X_val_preprocessed)
        
        # 각 폴드별 평가 지표 계산
        accuracy_scores.append(accuracy_score(y_val, y_val_pred))
        precision_scores.append(precision_score(y_val, y_val_pred))
        recall_scores.append(recall_score(y_val, y_val_pred))
        f1_scores.append(f1_score(y_val, y_val_pred))
    
    print(f"Fold Accuracy: {np.mean(accuracy_scores):.4f}, Precision: {np.mean(precision_scores):.4f}, "
          f"Recall: {np.mean(recall_scores):.4f}, F1-score: {np.mean(f1_scores):.4f}")
    
    return np.mean(accuracy_scores)

# 탐색할 하이퍼파라미터 범위 설정
pbounds = {
    'n_estimators': (100, 500),
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.6, 0.9),
    'colsample_bytree': (0.6, 0.9),
    'reg_alpha': (0, 1),
    'reg_lambda': (0, 1),
    'min_child_weight': (1, 10)
}

# Bayesian Optimization 객체 생성
optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds=pbounds,
    random_state=42
)

# 최적화 실행
optimizer.maximize(init_points=5, n_iter=20)

# 최적 하이퍼파라미터로 모델 재학습
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_child_weight'] = int(best_params['min_child_weight'])

# 최적 파라미터로 모델 설정
model = XGBClassifier(
    **best_params,
    random_state=42,
    eval_metric='logloss',
    tree_method='hist',
    device='cuda:0',
)

# 전체 학습 데이터로 모델 학습
X_train_preprocessed = preprocessor.fit_transform(X)
model.fit(X_train_preprocessed, y)

# 테스트 데이터 전처리 및 예측
X_test_preprocessed = preprocessor.transform(X_test)
y_test_pred = model.predict(X_test_preprocessed)

# 최종 학습 데이터에 대한 교차 검증 성능 출력
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, val_index in kf.split(X):
    X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
    y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # 전처리 적용
    X_tr_preprocessed = preprocessor.fit_transform(X_tr)
    X_val_preprocessed = preprocessor.transform(X_val)
    
    # 모델 학습
    model.fit(X_tr_preprocessed, y_tr)
    
    # 검증 데이터로 예측
    y_val_pred = model.predict(X_val_preprocessed)
    
    # 각 폴드별 평가 지표 계산
    accuracy_scores.append(accuracy_score(y_val, y_val_pred))
    precision_scores.append(precision_score(y_val, y_val_pred))
    recall_scores.append(recall_score(y_val, y_val_pred))
    f1_scores.append(f1_score(y_val, y_val_pred))

print("\n최종 모델의 교차 검증 성능:")
print(f"Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Precision: {np.mean(precision_scores):.4f}")
print(f"Recall: {np.mean(recall_scores):.4f}")
print(f"F1-score: {np.mean(f1_scores):.4f}")

# sample_submission.csv 파일 읽기
submit = pd.read_csv('sample_submission.csv')

# 예측 결과로 '허위매물여부' 컬럼 업데이트
submit['허위매물여부'] = y_test_pred

# 업데이트된 결과를 새로운 파일로 저장
submit.to_csv('fullk_fold_xgb_knn_bayesian_submission.csv', index=False)
print("예측 결과가 'fullk_fold_xgb_knn_bayesian_submission.csv' 파일로 저장되었습니다.")