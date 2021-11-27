import torch
from model import hybrid
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json


def a(sub_path):
    data_transform = transforms.Compose(
     [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img = Image.open(sub_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r',encoding='utf-8')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # create model
    model = net = hybrid(num_classes=10)
    # load model weights
    model_weight_path = "./hybrid.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    return class_indict[str(predict_cla)], predict[predict_cla].item()
    # plt.show()