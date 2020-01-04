import torch

torch.manual_seed(0)
a = torch.rand(2, 2)
# print(type(a.numpy()))
print(a.view(4))

print(a)

b = torch.ones(2, 2)

c = torch.add(a, b)
print(c)
c.add_(b)
print(c)
c.sub_(a)
print(c)
c.mul_(a)
print(c)
c.div_(a)
print(c)

print(a.mean(dim=0))

print(a.mean(dim=1))

print(a.std(dim=1))

x = torch.ones(2, requires_grad=True)
y = 5 * (x + 1) ** 2
print(y)
o = torch.mean(y)
print(o)
o.backward()
#o.backward(torch.tensor([3.0, 2.0], dtype=torch.float))
print()

print(x.grad)

