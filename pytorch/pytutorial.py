import torch

x = torch.ones(2, 2, requires_grad=True)

y = x + 2

print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z)
print(out)

a = torch.rand(2, 2)

a = ((a * 3) / (a - 1))

print(a)
print(a.requires_grad)

a.requires_grad_(True)
print(a)

b = (a * a).sum()
print(b.grad_fn)

out.backward()

print(x.grad)

c = torch.randn(3, requires_grad=True)

d = c * 2
while d.data.norm() < 1000:
    d = d * 2
print(d)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)

d.backward(v)

print(c.grad)

print(x.requires_grad)

print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
