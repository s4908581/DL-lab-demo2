import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as transforms
import time
from model import create_model  # 假设模型定义在model.py中
import os

def main():
    # 分布式训练初始化
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    # 超参数配置
    batch_size = 1024  # 大batch size充分利用A100
    epochs = 60  # 总epoch数
    base_lr = 0.4  # 基础学习率（OneCycle会调整）
    
    # 数据增强和预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 添加随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.RandomGrayscale(p=0.1),  # 随机灰度化
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # 随机擦除
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 分布式数据加载
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 创建模型并转为DDP
    model = create_model().to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    
    # # OneCycle学习率调度器
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=base_lr, steps_per_epoch=len(train_loader), epochs=epochs
    # )
    # 使用余弦退火学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 训练循环
    total_start = time.time()
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # 每100个batch打印一次
            if i % 100 == 0 and local_rank == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 评估模型
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
        
        # 保存模型
        torch.save(model.module.state_dict(), 'fast_cifar_model.pth')

if __name__ == '__main__':

    main()





