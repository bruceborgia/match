from mydataloader import CatImageDataset
from mynet import myresnet
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 根据需要调整大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imgdir = "./dataset"
    label_file = "./dataset/train_list.txt"

    dataset = CatImageDataset(image_dir=imgdir, label_file=label_file, transform=transform)

    batch_size = 8
    shuffle = True
    num_workers = 1

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    epochs = 200
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    net = myresnet().cuda(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_loss = []
    minloss = 100000
    print("start train")
    for epoch in range(epochs):
        net.train()
        sum_loss = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = net(x)

            optimizer.zero_grad()
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            train_loss.append(loss.item())

        print("epochs:%d, loss:%.3f" %(epoch, loss.item()))
        if loss.item() < minloss:
            torch.save(net.state_dict(), "./best.pth")
            minloss = loss.item()
        if epoch == 199:
            torch.save(net.state_dict(), "./last.pth")

