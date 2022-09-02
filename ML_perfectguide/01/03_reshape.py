import numpy as np

# p21
arr1 = np.arange(10)
print(arr1)
'''
[0 1 2 3 4 5 6 7 8 9]
'''
arr2 = arr1.reshape(-1, 5) # 고정된 열의 갯수인 5에 맞게 행을 자동으로 변환
print(arr2, arr2.shape)

'''
[[0 1 2 3 4]
 [5 6 7 8 9]] (2, 5)
'''
arr2 = arr1.reshape(5, -1) # 고정된 행의 갯수인 5에 맞게 열을 자동으로 변환
print(arr2, arr2.shape)
'''
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]] (5, 2)
'''

'''
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr.shape)

결과값: (2, 2, 3)

※ 2 : 1차원 요소 개수.
※ 2 : 2차원 각 차원의 요소 개수.
※ 3 : 3차원 각 차원의 요소 개수. 
'''

# p22
array1 = np.arange(8)
array3d = array1.reshape((2,2,2))
print('array3d:\n',array3d.tolist())
print(array1)
print(array3d)
'''
array3d:
 [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
[0 1 2 3 4 5 6 7]
[[[0 1]
  [2 3]]

 [[4 5]
  [6 7]]]
'''
# 3차원 ndarray를 2차원 ndarray로 변환하되 칼럼갯수는 1
array5 = array3d.reshape(-1, 1)
print('array5:\n',array5.tolist())
print('array5 shape:',array5.shape)

# 1차원 ndarray를 2차원 ndarray로 변환화되 칼럼 갯수는 1
array6 = array1.reshape(-1, 1)
print('array6:\n',array6.tolist())
print('array6 shape:',array6.shape)