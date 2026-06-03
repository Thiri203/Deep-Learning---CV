import torch
import torch.nn as nn
import copy
import time
from tqdm.auto import tqdm


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10, weights_name='best_model', is_inception=False):
    """
    Train a PyTorch model and save the best weights based on validation accuracy.

    Returns:
        model            : best model weights
        val_acc_history  : validation accuracy per epoch
        loss_history     : training loss per epoch
        epoch_times      : time taken per epoch (seconds)
    """
    device = next(model.parameters()).device

    best_weights = copy.deepcopy(model.state_dict())
    best_acc     = 0.0

    val_acc_history = []
    loss_history    = []
    epoch_times     = []

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        epoch_start = time.time()
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        # Each epoch has train and val phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss     = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(loader, desc=phase, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        # GoogLeNet returns 3 outputs during training
                        outputs, aux1, aux2 = model(inputs)
                        loss = (criterion(outputs, labels) +
                                0.3 * criterion(aux1, labels) +
                                0.3 * criterion(aux2, labels))
                    else:
                        outputs = model(inputs)
                        loss    = criterion(outputs, labels)

                    preds = outputs.argmax(dim=1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc  = running_corrects / len(loader.dataset)

            print(f'{phase} loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}')

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                loss_history.append(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc     = epoch_acc
                    best_weights = copy.deepcopy(model.state_dict())
                    torch.save(best_weights, f'{weights_name}.pth')
                    print(f'  -> Best model saved: {weights_name}.pth')

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f'Epoch time: {epoch_time:.1f}s')

    print(f'\nBest val accuracy: {best_acc:.4f}')
    model.load_state_dict(best_weights)
    return model, val_acc_history, loss_history, epoch_times


def test_model(model, test_loader):
    """Evaluate model on test set and return accuracy."""
    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total   = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs  = inputs.to(device)
            labels  = labels.to(device)
            outputs = model(inputs)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    acc = correct / total
    print(f'Test accuracy: {acc:.4f} ({acc*100:.2f}%)')
    return acc