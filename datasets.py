import os
import json
import cv2
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
import matplotlib.pyplot as plt


# all classes except road(driviable, 1), sidewalk, parking(non-drivibale, 2) are background(0)
def convert(gt_path):
    convert_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 2, 10: 2, 11: 0, 12: 0,
                    13: 0, 14: 0, 15: 0, 16: 0, 17: 5, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0,
                    24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, -1: 0}

    gt = cv2.imread(gt_path, 0)
    converted_gt = np.zeros_like(gt)

    for key in convert_dict.keys():
        index = np.where(gt == key)
        converted_gt[index] = convert_dict[key]

    converted_gt_dir = gt_path.split('.')[0] + 'new' + '.' + gt_path.split('.')[1]
    cv2.imwrite(converted_gt_dir, converted_gt)


def label_conversion(gt_dir):
    cities = os.listdir(gt_dir)
    print("converting...")
    for city in cities:
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in os.listdir(city_gt_dir):
            prefix = basename.split('_')[:3]
            gt_name = prefix[0] + '_' + prefix[1] + '_' + prefix[2] + '_' + "gtFine_labelIds.png"
            gt_path = os.path.join(city_gt_dir, gt_name)
            convert(gt_path)
    print("conversion finished")


def get_files(img_dir, gt_dir):
    files = []
    cities = os.listdir(img_dir)
    for city in cities:
        city_img_dir = os.path.join(img_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in os.listdir(city_img_dir):
            prefix = basename.split('_')[:3]
            # only for leftImg. For rightImg, modify the suffix.
            img_name = prefix[0] + '_' + prefix[1] + '_' + prefix[2] + '_' + "leftImg8bit.png"
            instance_name = prefix[0] + '_' + prefix[1] + '_' + prefix[2] + '_' + "gtFine_instanceIds.png"
            label_name = prefix[0] + '_' + prefix[1] + '_' + prefix[2] + '_' + "gtFine_labelIdsnew.png"
            json_name = prefix[0] + '_' + prefix[1] + '_' + prefix[2] + '_' + "gtFine_polygons.json"

            img_file = os.path.join(city_img_dir, img_name)
            instance_file = os.path.join(city_gt_dir, instance_name)
            label_file = os.path.join(city_gt_dir, label_name)
            json_file = os.path.join(city_gt_dir, json_name)

            files.append((img_file, instance_file, label_file, json_file))

    return files


def load_sem_seg(img_dir, gt_dir):
    ret = []
    for img_file, _, label_file, json_file in get_files(img_dir, gt_dir):
        with open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret.append(
            {
                "file_name": img_file,
                "sem_seg_file_name": label_file,
                "height": jsonobj["imgHeight"],
                "width": jsonobj["imgWidth"],
            }
        )

    return ret


class Cityscapes(data.Dataset):
    def __init__(self, lst):
        self.lst = lst
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        image = cv2.imread(self.lst[index]["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.lst[index]["sem_seg_file_name"], 0)

        image = cv2.resize(image, (224, 224), cv2.INTER_LINEAR)
        image = self.transforms(image)
        gt = cv2.resize(gt, (224, 224), cv2.INTER_LINEAR)
        gt = T.Compose([T.ToTensor()])(gt)
        gt = torch.squeeze(gt)
        gt = gt.type(torch.LongTensor)

        return image, gt

    def __len__(self):
        return len(self.lst)

# # uncomment it for test
# if __name__ == "__main__":
#     lst = load_sem_seg(img_dir="/home/julius/leftImg8bit_trainvaltest/leftImg8bit/train",
#     gt_dir="/home/julius/gtFine_trainvaltest/gtFine/train")
#     dataset = Cityscapes(lst, transform=None)
#
#     fig = plt.figure()
#
#     for i in range(len(dataset)):
#         img, gt = dataset[i]
#         ax = plt.subplot(1, 4, i + 1)
#         plt.tight_layout()
#         ax.set_title('Sample #{}'.format(i))
#         ax.axis('off')
#         plt.imshow(img)
#
#         if i == 3:
#             plt.show()
#             break

