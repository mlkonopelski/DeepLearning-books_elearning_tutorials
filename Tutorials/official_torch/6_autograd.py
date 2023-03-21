import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

print(f'w values: {w}')
print(f'b values {b}')

z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss.backward()

print(f'w gradient values: {w.grad}')
print(f'b gradient values {b.grad}')