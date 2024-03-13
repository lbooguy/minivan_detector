import torch
from PIL import Image
from torchvision.io.image import read_image
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
import easyocr
import cv2
LABELS = {1: 'Minivan'}

image_path = "C:/Users/USER/Downloads/testMinivan/minivan_4.jpg"
img = Image.open(image_path)

readeren = easyocr.Reader(['en'])
readerru = easyocr.Reader(['ru'])

device = torch.device("cpu")
model = torch.load(
    "C:/Users/USER/PycharmProjects/UniWork/MinivanDetection/trainedmodels/ssd_15_epochs.pt",
    map_location=torch.device('cpu'))
model.score_thresh = 0.5
model = model.to('cpu')
model.eval()

w = SSD300_VGG16_Weights.DEFAULT
preprocess = w.transforms()
batch = [preprocess(img)]
prediction = model(batch)[0]
image = read_image(image_path)

num_of_pred = int(prediction['labels'].shape[0])
labels = []
if prediction['boxes'].shape[0] != 0:
    for label in range(num_of_pred):
        '''
        imgcv = cv2.imread(image_path)
        crop_img = imgcv[int(prediction['boxes'][label][1]):int(prediction['boxes'][label][3]), int(prediction['boxes'][label][0]):int(prediction['boxes'][label][2])]
        texten = readeren.readtext(crop_img, detail=0)
        textru = readerru.readtext(crop_img, detail=0)
        print("Распознанный русский текст на изображении", textru)
        print("Распознанный английский текст на изображении", texten)
        labels.append(
            f"{LABELS[int(prediction['labels'][label])]} {100 * round(float(prediction['scores'][label]), 3)} \nen:{texten} \nru:{textru}")
        '''
        labels.append(
            f"{LABELS[int(prediction['labels'][label])]} {100 * round(float(prediction['scores'][label]), 3)}")

    box = draw_bounding_boxes(image, boxes=prediction["boxes"],
                              labels=labels,
                              colors="red",
                              width=4,
                              font='C:/Windows/Fonts/Arial.ttf',
                              font_size=20)
else:
    box = draw_bounding_boxes(image, boxes=torch.tensor([[0, 0, 0, 0]]),
                              labels=['image have no minivans'],
                              colors="red",
                              width=4,
                              font='C:/Windows/Fonts/Arial.ttf',
                              font_size=55)

im = to_pil_image(box.detach())
im.show()
