import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

titanic_df = pd.read_csv('02/data/train.csv')
# ic(titanic_df.head(3))

# print('\n ### train 데이터 정보 ###  \n')
# print(titanic_df.info())

titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)

# print('데이터 세트 Null 값 갯수 ',titanic_df.isnull().sum().sum())


# object 컬럼타입 추출
titanic_df.dtypes[titanic_df.dtypes == 'object'].index.tolist()
'''['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']'''


# print(' Sex 값 분포 :\n',titanic_df['Sex'].value_counts())
# print('\n Cabin 값 분포 :\n',titanic_df['Cabin'].value_counts())
# print('\n Embarked 값 분포 :\n',titanic_df['Embarked'].value_counts())
'''
 Sex 값 분포 :
 male      577
female    314
Name: Sex, dtype: int64

 Cabin 값 분포 :
 N              687
C23 C25 C27      4  -> 한 꺼번에 표기된 cabin 값도 존재
G6               4
B96 B98          4
C22 C26          3
              ...
E34              1
C7               1
C54              1
E36              1
C148             1
Name: Cabin, Length: 148, dtype: int64

 Embarked 값 분포 :
 S    644
C    168
Q     77
N      2
Name: Embarked, dtype: int64
'''
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].head(3)) # 선실 등급인 첫번째 알파벳만 추출
'''
0    N
1    C
2    N
'''