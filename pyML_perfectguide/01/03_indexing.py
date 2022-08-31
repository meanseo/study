import numpy as np

arr = np.arange(18)
arr = arr.reshape(2,3,3)
print(arr)
print(arr[1,1,2])
'''
[[[ 0  1  2]  
  [ 3  4  5]  
  [ 6  7  8]] 

 [[ 9 10 11]  
  [12 13 14]  
  [15 16 17]]]
14
'''