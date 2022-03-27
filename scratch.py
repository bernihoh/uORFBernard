import torch
from torch import nn

#loss = nn.MSELoss()
input = torch.randn(4, 5, requires_grad=True)
target = torch.randn(10, 5)
output = torch.cdist(input, target, 2)
#output.backward()
print(input[0].shape)
print(target)
print(output)

min_output, lbl = torch.min(output, dim=0)
print(min_output, lbl, lbl.shape)
maxi = torch.max(min_output).type(torch.int)
print(maxi)
print(torch.cat((output, min_output[None, ...]), dim=0))

print(target[torch.randint(0, target.shape[0], (1,))].shape)

used_lbls = torch.arange(torch.max(lbl)+1).view(torch.max(lbl)+1, 1)
lbl_mask = used_lbls.repeat(1, lbl.shape[0])
print(used_lbls)
print(lbl_mask, lbl_mask.shape)
lbl_mask = torch.subtract(lbl_mask, lbl)
print(lbl_mask)
lbl_mask = lbl_mask.eq(0)#.type(torch.int)
print(lbl_mask, lbl_mask.shape)
lbl_sum = torch.sum(lbl_mask, dim=1).view(torch.max(lbl)+1, 1)
print(lbl_sum)
target = target.repeat(torch.max(used_lbls)+1, 1, 1)
print(target.shape, lbl_mask.shape)
einsum = torch.einsum('abc,ab->abc', target, lbl_mask)
print(einsum, einsum.shape)
lbl_einsum_sum = torch.sum(einsum, dim=1)
print(lbl_einsum_sum, lbl_einsum_sum.shape)
mean_sum = torch.divide(lbl_einsum_sum, lbl_sum)
print(mean_sum, mean_sum.shape)
mean_sum = mean_sum[[~torch.any(mean_sum.isnan(), dim=1)]]
print(mean_sum, mean_sum.shape)


