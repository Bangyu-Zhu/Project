import torch
import cv2
from model import ResUnet
from datasets import Cityscapes, label_conversion, load_sem_seg
from torch.utils.data import DataLoader


def inference(test_img_dir, test_gt_dir):
    net = ResUnet('resnet34')
    net.load_state_dict(torch.load('./resunet.pth'))
    net.eval()
    test_file_list = load_sem_seg(test_img_dir, test_gt_dir)
    test_dataset = Cityscapes(test_file_list)
    testloader = DataLoader(test_dataset, batch_size=1,
                            shuffle=False, num_workers=8, pin_memory=True)
    img, label = next(iter(testloader))
    pred = net(img)
    pred = pred.squeeze().detach().numpy()

    print(pred)


if __name__ == "__main__":
    test_img_dir = "/disk/users/hb662/datasets/leftImg8bit_trainvaltest/leftImg8bit/test"
    test_gt_dir = "/disk/users/hb662/datasets/gtFine_trainvaltest/gtFine/test"
    inference(test_img_dir, test_gt_dir)