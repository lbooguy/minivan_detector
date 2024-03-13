import torch
import torchvision.transforms.functional
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from os import listdir, path


class TRANSFORM(object):
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        image = torchvision.transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
        return image, target


# xml_paths = 'C:/Users/USER/PycharmProjects/UniWork/MinivanDetection/Dataset/van/pascal'
# img_dir = 'C:/Users/USER/PycharmProjects/UniWork/MinivanDetection/Dataset/van/images'

LABELS = {'van': 1}


class MinivanDataset(Dataset):
    def __init__(self, xml_paths, img_dir, transforms=TRANSFORM()):
        self.xml_paths = xml_paths
        self.xml_files = list(sorted(listdir(path.join(xml_paths))))
        self.img_dir = img_dir
        self.images = list(sorted(listdir(path.join(img_dir))))
        self.transforms = transforms
        self.data = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        xml_path = path.join(self.xml_paths, self.xml_files[idx])
        img_path = path.join(self.img_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, area, labels,iscrowd = [], [], [], []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            area.append((ymax - ymin) * (xmax - xmin))
            labels.append(LABELS[name])
            iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        image_id = torch.tensor([idx])
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
