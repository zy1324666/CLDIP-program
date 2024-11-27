import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, roc_curve, auc, r2_score
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd  # 导入 pandas 库



res = loadmat('I:\\学习文件\\A论文相关\\C文章撰写中\\物理导向性边坡稳定性分析模型\\Modle1.mat')   

data_key = next(key for key in res.keys() if not key.startswith('__'))
data = res[data_key]


X = data[:, :-1]  
y = data[:, -1]   

scaler = StandardScaler()
X = scaler.fit_transform(X)


permutation = np.random.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
}


kf = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(kf.split(X)):
 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    train_data = lgb.Dataset(X_train, y_train)
    test_data = lgb.Dataset(X_test, y_test)


    print(f'开始交叉验证 {i + 1}...')
    gbm = lgb.train(params,
                    train_set=train_data,
                    num_boost_round=100,
                    valid_sets=[train_data, test_data])


    y_pred_proba = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)


    confu = np.zeros((2, 2))
    for j in range(len(y_pred_proba)):
        box = 1 if y_pred_proba[j] >= 0.5 else 0
        if box == 0 and y_test[j] == 0:
            confu[0, 0] += 1  # TN
        elif box == 0 and y_test[j] == 1:
            confu[0, 1] += 1  # FN
        elif box == 1 and y_test[j] == 0:
            confu[1, 0] += 1  # FP
        elif box == 1 and y_test[j] == 1:
            confu[1, 1] += 1  # TP

    TN, FP, FN, TP = confu.ravel()
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    Accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    Pp = (TP + TN) / (TP + TN + FP + FN)
    Pe = ((TP + FN) * (TP + FP) + (TN + FN) * (TN + FP)) / ((TP + TN + FP + FN) ** 2)
    Kappa = (Pp - Pe) / (1 - Pe) if (1 - Pe) > 0 else 0


    with pd.ExcelWriter(f'Cross_{i + 1}_results动力.xlsx') as writer:
        df_fprc = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        df_fprc.to_excel(writer, sheet_name='FPR_TPR', index=False)

        df_predictions = pd.DataFrame({'Y_pred': y_pred_proba, 'Y_true': y_test})
        df_predictions.to_excel(writer, sheet_name='Predictions', index=False)

        df_metrics = pd.DataFrame({
            'Precision': [Precision],
            'Recall': [Recall],
            'F1_Score': [F1],
            'Accuracy': [Accuracy],
            'Kappa': [Kappa],
            'AUC': [roc_auc]
        })
        df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
    print(f'AUC: {roc_auc}')

    mse = mean_squared_error(y_test, y_pred_proba)
    r2 = r2_score(y_test, y_pred_proba)

    rmse = np.sqrt(mse)

    print(f'交叉 {i + 1} 的 MSE: {mse:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}')

##################  Control data available   ###################
res = loadmat('I:\\学习文件\\A论文相关\\C文章撰写中\\物理导向性边坡稳定性分析模型\\鲁甸地震\\C数据\\NModle1_jy.mat') 

data_key = next(key for key in res.keys() if not key.startswith('__'))
data = res[data_key]

X_predict = data[:, :-1] 
y_predict = data[:, -1]  


scaler = StandardScaler()
X_predict = scaler.fit_transform(X_predict)


predict_data = lgb.Dataset(X_predict, y_predict)


y_pred_proba = gbm.predict(X_predict, num_iteration=gbm.best_iteration)
y_pred_proba = y_pred_proba
fpr, tpr, _ = roc_curve(y_predict, y_pred_proba)
roc_auc = auc(fpr, tpr)


confu = np.zeros((2, 2))
for j in range(len(y_pred_proba)):
    box = 1 if y_pred_proba[j] >= 0.5 else 0
    if box == 0 and y_predict[j] == 0:
        confu[0, 0] += 1  # TN
    elif box == 0 and y_predict[j] == 1:
        confu[0, 1] += 1  # FN
    elif box == 1 and y_predict[j] == 0:
        confu[1, 0] += 1  # FP
    elif box == 1 and y_predict[j] == 1:
        confu[1, 1] += 1  # TP

TN, FP, FN, TP = confu.ravel()
Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
F1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
Accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
Pp = (TP + TN) / (TP + TN + FP + FN)
Pe = ((TP + FN) * (TP + FP) + (TN + FN) * (TN + FP)) / ((TP + TN + FP + FN) ** 2)
Kappa = (Pp - Pe) / (1 - Pe) if (1 - Pe) > 0 else 0

print(f'/n')
print(f'AUC: {roc_auc}')
print(f'Precision: {Precision:.6f}, Recall: {Recall:.6f}, F1 Score: {F1:.6f}, Accuracy: {Accuracy:.6f}, Kappa: {Kappa:.6f}')


df_roc = pd.DataFrame({
    'False Positive Rate (FPR)': fpr,
    'True Positive Rate (TPR)': tpr
})


df_metrics = pd.DataFrame({
    'Metric': ['TN', 'FP', 'FN', 'TP', 'Precision', 'Recall', 'F1', 'Accuracy', 'Kappa', 'AUC', 'Pp', 'Pe'],
    'Value': [TN, FP, FN, TP, Precision, Recall, F1, Accuracy, Kappa, roc_auc, Pp, Pe]
})

df_combined = pd.concat([df_roc, df_metrics], axis=1)

df_combined.to_csv('model_2.csv', index=False)

mse = mean_squared_error(y_predict, y_pred_proba)
r2 = r2_score(y_predict, y_pred_proba)

rmse = np.sqrt(mse)

print(f'MSE: {mse:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}')

y_pred = (y_pred_proba >= 0.5).astype(int)
预测结果 = pd.DataFrame({'Predicted': y_pred, 'Probabilities': y_pred_proba})
预测结果.to_csv('预测结果.csv', index=False)
print("预测结果已保存")