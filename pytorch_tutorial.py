import torch 
import numpy as np

x=torch.rand(2,2,3)
print(x)


y=torch.ones(2,2,dtype=torch.int)
print(y)

z=torch.tensor([2.5,0.1])
print(z)

w=torch.rand(5,3)
print(w[1,:])

a=np.ones(5)
#print(a)
#b=a.numpy()
b=torch.from_numpy(a)
print(b)
a+=1
print(b)