

import torch
from sklearn.metrics import roc_auc_score
import torchvision



model = # load model

# Load Cifar10 dataset
in_distribution_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform= #depends on the model
)

# SVHN
out_of_distribution_dataset = torchvision.datasets.SVHN(
    root='./data',
    split='test',
    download=True,
    transform= #depends on the model
)



inputs = torch.cat([in_distribution_dataset.inputs,
                    out_of_distribution_dataset.inputs], dim=0)
targets = torch.cat([torch.zeros(in_distribution_dataset.inputs.size(0)),
                    torch.ones(out_of_distribution_dataset.inputs.size(0))], dim=0)


tensor_dataset = torch.utils.data.TensorDataset(inputs, targets)
data_loader = torch.utils.data.DataLoader(
    dataset=tensor_dataset,
    batch_size=# ,
    shuffle=False,
    num_workers=# ,
    pin_memory=# 
)

entropies = []


for inputs, targets in data_loader:
    inputs = inputs.to(device='cuda')
    targets = targets.to(device='cuda')

    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)

    entropies.append(
        -torch.sum(probs * torch.log(probs + 1e-10), dim=1).cpu()
    )

auc = roc_auc_score(
    targets.cpu().numpy(),
    torch.cat(entropies).cpu().numpy()
)