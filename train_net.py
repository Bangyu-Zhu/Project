import torch
import torch.optim as optim
import torch.nn as nn
from model import ResUnet
from datasets import Cityscapes, label_conversion, load_sem_seg
from torch.utils.data import DataLoader


def train(network, train_img_dir, train_gt_dir, val_img_dir, val_gt_dir,
          lr, epochs, lbl_conversion=False):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = ResUnet(network)
    net.to(device).train()

    optimizer = optim.SGD(net.head.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    if lbl_conversion:
        label_conversion(train_gt_dir)
        label_conversion(val_gt_dir)

    train_file_list = load_sem_seg(train_img_dir, train_gt_dir)
    train_dataset = Cityscapes(train_file_list)
    trainloader = DataLoader(train_dataset, batch_size=25,
                            shuffle=True, num_workers=8, pin_memory=True)

    val_file_list = load_sem_seg(val_img_dir, val_gt_dir)
    val_dataset = Cityscapes(val_file_list)
    valloader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=8, pin_memory=True)
    check_point_path = './checkpoints/'

    print('Begin of the Training')
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, data in enumerate(trainloader):
            optimizer.zero_grad()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            print(epoch, idx, loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, idx + 1, running_loss / 2000))
                running_loss = 0.0

    print('End of the Training')

    PATH = './resunet.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == "__main__":
    train_img_dir = "/disk/users/hb662/datasets/leftImg8bit_trainvaltest/leftImg8bit/train"
    train_gt_dir = "/disk/users/hb662/datasets/gtFine_trainvaltest/gtFine/train"
    val_img_dir = "/disk/users/hb662/datasets/leftImg8bit_trainvaltest/leftImg8bit/val"
    val_gt_dir = "/disk/users/hb662/datasets/gtFine_trainvaltest/gtFine/val"
    # for the first time, change lbl_conversion to True to convert labels, it will take about 30 minutes
    train("resnet34", train_img_dir, train_gt_dir, val_img_dir, val_gt_dir, 0.0001, 5, lbl_conversion=False)