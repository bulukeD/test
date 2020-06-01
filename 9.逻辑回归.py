# https://archive.ics.uci.edu/ml/machine-learning-databases/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def logistic():
    '''
    逻辑回归预进行癌症预测
    :return: None
    '''

    # 读取数据
    data_address = r'E:\机器学习\机器学习\数据\数据2.xlsx'
    data = pd.read_excel(data_address)
    data = data.iloc[:,1:]
    # print(data)

    # 进行缺失值处理
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna()
    # print(data)

    # 进行数据集的分割
    x = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)
    print(lg.coef_)
    y_predict = lg.predict(x_test)
    print('逻辑回归预测的准确率：', lg.score(x_test, y_test))
    print('逻辑回归预测的召回率：', classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性']))

    return None


if __name__=='__main__':
    logistic()