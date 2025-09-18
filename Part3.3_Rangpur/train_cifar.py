import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as transforms
import time
from model import create_model
import os

def main():
    # Initialization 
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    # Parameter
    batch_size = 1024 
    epochs = 60
    base_lr = 0.4
    
    # Data Preprocessing and augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomGrayscale(p=0.1), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Data loader
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    #====================================
    # Create model
    model = create_model().to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    
    # OneCycle LR schedular
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=base_lr, steps_per_epoch=len(train_loader), epochs=epochs
    )
    
    # Mixed-precision training
    scaler = GradScaler()
    #====================================    
    # Training
    total_start = time.time()
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            
            if i % 100 == 0 and local_rank == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
        #=============================
        # Evaluation
        if local_rank == 0:
            epoch_time = time.time() - epoch_start
            print(f'Epoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f}s')
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Test Accuracy: {accuracy:.2f}%')
    
    total_time = time.time() - total_start
    if local_rank == 0:
        print(f'Total training time: {total_time:.2f}s')
        print(f'Final accuracy: {accuracy:.2f}%')
        
        
        torch.save(model.module.state_dict(), 'fast_cifar_model.pth')

if __name__ == '__main__':

    main()







