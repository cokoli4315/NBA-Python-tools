import torch

x=torch.randn(3,requires_grad=True)
print(x)

y=x+2

print(y)
z=y*y*2
z=z.mean()
print(z)

z.backward()
print(x.grad)

with torch.no_grad():
    y=x+2
    print(y)


weights = torch.ones(4,requires_grad=True)
for epoch in range(3):
    model_output=(weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()

    #optimizer= torch.optim.SGD(weights,lr=0.01)
    #optimizer.step()
    #optimizer.zero_grad()


    #requires grad must be specified when calculating gradiant 

    #backward()
    #must call grad_zero

x=torch.tensor(1.0)
y=torch.tensor(2.0)
w=torch.tensor(1.0,requires_grad=True)

#forward pass and compute the loss
y_hat=w*x
loss=(y_hat-y)**2
print(loss)

#backward pass
loss.backward()
print(w.grad)

#update weights
#next forward 

