# 행렬 연습
# [ ] 개수대로 거슬러 올라가서 제일 작은 [ ] 안에 있는 값이 열이 된다.
# 그 다음부터 순차적으로 열 행 다중행렬 다중텐서순이다.
import numpy as np 

a1 = np.array( [ [1,2], [3,4] , [5,6]] )
a2 = np.array( [ [1,2,3], [4,5,6] ] )
a3 = np.array( [ [[ [[1],[2],[3]] , [[1],[1],[1]]]] ] )
a4 = np.array( [ [[1,2], [3,4]] , [[5,6],[5,6]] ] )
a5 = np.array( [ [[1,2,3] , [4,5,6]] ])
a6 = np.array( [ 1,2,3,4,5] )

print(a1,a1.shape)
print(a2,a2.shape)
print(a3,a3.shape)
print(a4,a4.shape)
print(a5,a5.shape)
print(a6,a6.shape)