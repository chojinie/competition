import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 데이터 불러오기
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 범주형 데이터 인코딩
one_hot_cols = ['\ub9e4\ubb34\ud655\uc778\ubc29\uc2dd', '\ubc29\ud654', '\uc8fc\ucc28\uac00\ub2a5\uc5ec\ubd80', '\uc911\uac00\uc0ac\ubb34\uc18c', '\uc81c\uc548\ud50c\ub79c\ud3fc']
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
train_encoded = one_hot_encoder.fit_transform(train_data[one_hot_cols]).toarray()
test_encoded = one_hot_encoder.transform(test_data[one_hot_cols]).toarray()

train_encoded_df = pd.DataFrame(train_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols), index=train_data.index)
test_encoded_df = pd.DataFrame(test_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols), index=test_data.index)

train_data = pd.concat([train_data.drop(columns=one_hot_cols), train_encoded_df], axis=1)
test_data = pd.concat([test_data.drop(columns=one_hot_cols), test_encoded_df], axis=1)

# 날짜 데이터 처리
train_data['\uac8c\uc7ac\uc77c'] = pd.to_datetime(train_data['\uac8c\uc7ac\uc77c'])
test_data['\uac8c\uc7ac\uc77c'] = pd.to_datetime(test_data['\uac8c\uc7ac\uc77c'])

train_data['\uac8c\uc7ac\ub144\ub3c4'] = train_data['\uac8c\uc7ac\uc77c'].dt.year
train_data['\uac8c\uc7ac\uc6d4'] = train_data['\uac8c\uc7ac\uc77c'].dt.month
train_data['\uac8c\uc7ac\uc77c\uc790'] = train_data['\uac8c\uc7ac\uc77c'].dt.day
train_data['\uac8c\uc7ac\uc6d4\uc694\uc77c'] = train_data['\uac8c\uc7ac\uc77c'].dt.weekday
train_data.drop(columns=['\uac8c\uc7ac\uc77c'], inplace=True)

test_data['\uac8c\uc7ac\ub144\ub3c4'] = test_data['\uac8c\uc7ac\uc77c'].dt.year
test_data['\uac8c\uc7ac\uc6d4'] = test_data['\uac8c\uc7ac\uc77c'].dt.month
test_data['\uac8c\uc7ac\uc77c\uc790'] = test_data['\uac8c\uc7ac\uc77c'].dt.day
test_data['\uac8c\uc7ac\uc6d4\uc694\uc77c'] = test_data['\uac8c\uc7ac\uc77c'].dt.weekday
test_data.drop(columns=['\uac8c\uc7ac\uc77c'], inplace=True)

# 수치형 데이터 스케일링 및 결측치 보완
numeric_columns = ['\ubcf4\uc99d\uae08', '\uc6d4\uc138', '\uc804\uc6a9\uba74\uc801', '\ud574\ub2f9\uce35', '\ucd1d\uce35', '\ubc29\uc218', '\uc694\uc2dc\uc218', '\ucd1d\uc8fc\ucc28\ub300\uc218']
columns_fill_knn = ['\uc804\uc6a9\uba74\uc801', '\ud574\ub2f9\uce35', '\ucd1d\uce35', '\ubc29\uc218', '\uc694\uc2dc\uc218', '\ucd1d\uc8fc\ucc28\ub300\uc218']

scaler = StandardScaler()
knn_imputer = KNNImputer(n_neighbors=7)

train_data[numeric_columns] = knn_imputer.fit_transform(train_data[numeric_columns])
train_data[numeric_columns] = scaler.fit_transform(train_data[numeric_columns])

test_data[numeric_columns] = knn_imputer.transform(test_data[numeric_columns])
test_data[numeric_columns] = scaler.transform(test_data[numeric_columns])

# 특성과 레이블 분리
X = train_data.drop(columns=['ID', '\ud5c8\uc704\ub9e4\ubb34\uc5ec\ubd80']).values
y = train_data['\ud5c8\uc704\ub9e4\ubb34\uc5ec\ubd80'].values

X_test = test_data.drop(columns=['ID']).values

# PyTorch DNN 모델 정의
class DNNModel(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super(DNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# K-Fold Cross Validation 및 성능 측정
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []

for train_index, val_index in kf.split(X):
    X_tr, X_val = X[train_index], X[val_index]
    y_tr, y_val = y[train_index], y[val_index]

    # 데이터 텐서로 변환
    train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 모델 생성 및 학습 설정
    model = DNNModel(input_dim=X.shape[1], dropout_rate=0.3)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 루프
    for epoch in range(100):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # 검증
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                val_preds.extend(model(X_batch).squeeze().numpy())
        val_preds = np.array(val_preds) > 0.5
        accuracy_scores.append(accuracy_score(y_val, val_preds))

print(f"DNN 모델의 평균 Accuracy: {np.mean(accuracy_scores):.4f}")

# 최적의 하이퍼파라미터 탐색을 위한 Bayesian Optimization 설정
def dnn_cv(dropout_rate, learning_rate):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []

    for train_index, val_index in kf.split(X):
        X_tr, X_val = X[train_index], X[val_index]
        y_tr, y_val = y[train_index], y[val_index]

        train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = DNNModel(input_dim=X.shape[1], dropout_rate=dropout_rate)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(50):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                val_preds.extend(model(X_batch).squeeze().numpy())
        val_preds = np.array(val_preds) > 0.5
        accuracy_scores.append(accuracy_score(y_val, val_preds))

    return np.mean(accuracy_scores)

pbounds = {
    'dropout_rate': (0.1, 0.5),
    'learning_rate': (0.0001, 0.01)
}

optimizer = BayesianOptimization(
    f=dnn_cv,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=5, n_iter=20)

best_params = optimizer.max['params']

# 최적 파라미터로 최종 모델 학습
final_model = DNNModel(input_dim=X.shape[1], dropout_rate=best_params['dropout_rate'])
criterion = nn.BCELoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])

train_dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(100):
    final_model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = final_model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

# 테스트 데이터 예측
test_tensor = torch.FloatTensor(X_test)
y_test_pred = (final_model(test_tensor).squeeze().detach().numpy() > 0.5).astype(int)

# 결과 저장
submit = pd.read_csv('sample_submission.csv')
submit['\ud5c8\uc704\ub9e4\ubb34\uc5ec\ubd80'] = y_test_pred
submit.to_csv('dnn_knn_bayesian_submission.csv', index=False)
print("예측 결과가 'dnn_knn_bayesian_submission.csv' 파일로 저장되었습니다.")
