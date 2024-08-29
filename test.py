import torch

p = torch.tensor([1. ,2., 3.], requires_grad=True)
b = torch.tensor([1.5], requires_grad=False)
a = torch.tensor([4. ,5. ,6.])
optimizer = torch.optim.Adam(params=[p], lr=0.01)
loss = ((p * b - a) ** 2).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(p)