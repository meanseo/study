from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from icecream import ic

# 데이터 로드
iris = load_iris()

# 피쳐와 타겟이 이미 나눠져 있음
iris_data = iris.data
iris_label = iris.target
print('iris target 값: ', iris_label)
'''
iris target 값:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''
print('iris target 명: ', iris.target_names)
'''
iris target 명:  ['setosa' 'versicolor' 'virginica']
setosa : 0
versicolor : 1
virginica : 2
'''

iris_df = pd.DataFrame(data= iris_data, columns=iris.feature_names)
iris_df['label'] = iris_label
ic(iris_df.head(5))
'''
ic| iris_df.head(5):    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  label
                     0                5.1               3.5                1.4               0.2      0
                     1                4.9               3.0                1.4               0.2      0
                     2                4.7               3.2                1.3               0.2      0
                     3                4.6               3.1                1.5               0.2      0
                     4                5.0               3.6                1.4               0.2      0
'''

x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)