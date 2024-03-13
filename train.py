import torch
import torchvision.utils
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights
from torchvision.models.detection.ssd import det_utils
import utils
from engine import train_one_epoch, evaluate
from DataLoader import MinivanDataset
from sklearn.model_selection import train_test_split
import gc

xml_paths = 'C:/Users/USER/PycharmProjects/UniWork/MinivanDetection/Dataset/van/pascal'
img_dir = 'C:/Users/USER/PycharmProjects/UniWork/MinivanDetection/Dataset/van/images'

if __name__ == '__main__':
    print('Begin of loading data')
    #del variables
    #gc.collect()

    dataset = MinivanDataset(
        xml_paths=xml_paths,
        img_dir=img_dir)

    dataset_train, dataset_valid = train_test_split(dataset, test_size=0.2, random_state=42)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=8, shuffle=True, #num_workers=2,
        collate_fn=utils.collate_fn)
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=8, shuffle=False, #num_workers=2,
        collate_fn=utils.collate_fn)
    print('End of loading data')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device is',device)
    torch.cuda.empty_cache()
    model = torchvision.models.detection.ssd300_vgg16(weights="SSD300_VGG16_Weights.COCO_V1")
    num_classes = 2
    num_anchors = model.anchor_generator.num_anchors_per_location()
    in_channels = det_utils.retrieve_out_channels(model.backbone, (300, 300))
    model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(in_channels=in_channels,
                                                                                            num_anchors=num_anchors,
                                                                                            num_classes=num_classes)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    print('Begin of train')
    num_epochs = 30
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        if epoch%12 == 0:
            optimizer = torch.optim.SGD(params, lr=0.0005,
                                        momentum=0.9, weight_decay=0.0005)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=3,
                                                           gamma=0.1)
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the valid dataset
        evaluate(model, data_loader_valid, device=device)
        if epoch % 5 ==0:
            torch.save(model, "C:/Users/USER/PycharmProjects/UniWork/MinivanDetection/trainedmodels/ssd_" + str(epoch) + "_epochs" + ".pt")
    torch.save(model, "C:/Users/USER/PycharmProjects/UniWork/MinivanDetection/trainedmodels/SSD_van.pt")

#C:\Users\USER\PycharmProjects\UniWork\MinivanDetection\trainedmodels