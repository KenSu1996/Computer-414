
import torch

from torch.utils.data import Subset

from sklearn.model_selection import train_test_split

class MetricLog():
    def __init__(self) -> None:
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        self.vlist = []

    def update(self, val):
        self.vlist.append(val)
        self.val = val
        self.sum = self.sum + val
        self.count = self.count + 1
        self.avg = self.sum/self.count


def train_epoch(model, train_loader, criterion, optimizer, criterion_optimizer, lr_scheduler, device):
    train_epoch_losses = MetricLog()
    for idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)
        model = model.to(device)
        optimizer.zero_grad()
        criterion_optimizer.zero_grad()
        preds_tuple = model(inputs)
        criterion = criterion.to(device, non_blocking = True)
        loss_criterion = criterion(preds_tuple[1], labels)
        loss = loss_criterion
        train_epoch_losses.update(loss.item())
        loss.backward()
        optimizer.step()
        criterion_optimizer.step()
    print('\nTraining set: Average loss: {}\n'.format(train_epoch_losses.avg), flush=True)
    return train_epoch_losses.avg





def val_epoch(model, val_loader, criterion, optimizer, device):
    val_losses = MetricLog()
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            inputs,labels = data
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            model = model.to(device)
            preds_tuple = model(inputs)
            loss_criterion = criterion(preds_tuple[1], labels)
            loss = loss_criterion
            val_losses.update(loss.item())
    print('\nValidation set: Average loss: {}\n'.format(val_losses.avg), flush=True)
    return val_losses.avg



def train_model(model, train_loader, val_loader, criterion, optimizer, criterion_optimizer, lr_scheduler, epochs, save_path, arch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(epochs):
        lr_scheduler.step()
        print('Epoch:', epoch,'LR:', lr_scheduler.get_lr())
        train_epoch(model, train_loader, criterion, optimizer, criterion_optimizer, lr_scheduler, device)
        val_loss = val_epoch(model, val_loader, criterion, optimizer, device)
        save_name = save_path + 'model_' + str(epoch+1) + '_checkpoint.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.state_dict(),
            'loss': val_loss,
        }, save_name)



def train_val_split(dataset, val_size = 0.3):
    train_idx, val_idx = train_test_split(list(range(len(dataset.targets))), test_size = val_size, stratify = dataset.targets)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    return train_dataset, val_dataset