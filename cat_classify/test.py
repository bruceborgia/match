from mydataloader import CatImageDataset
from mynet import myresnet
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
from PIL import Image

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 根据需要调整大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imgfolder = "./dataset/cat_12_test"
    # test_folder = "cat_12_test"

    imgs = os.listdir(imgfolder)
    test_data = {}
    for i in imgs:
        img_tensor =  transform(Image.open(os.path.join(imgfolder, i)).convert('RGB'))
        img_tensor = img_tensor.unsqueeze(0)
        test_data[i] = img_tensor

    result = list()
    net = myresnet()
    net.load_state_dict(torch.load("./best.pth"))
    net.eval()
    with torch.no_grad():
        for name, img in test_data.items():
            y_pred = net(img)
            label = torch.argmax(y_pred)
            result.append((name, label.item()))

    import csv

    # 假设result是您之前代码中生成的列表
    # result = [('image1.jpg', 0), ('image2.png', 1), ...]

    # 定义CSV文件的名称
    csv_filename = 'submit.csv'

    # 使用'with'语句打开文件，确保正确关闭
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        # 创建一个csv.writer对象
        writer = csv.writer(file)

        # 首先写入列标题（可选）
        # writer.writerow(['Image Name', 'Predicted Label'])

        # 然后遍历result列表，写入每一行
        for img_name, predicted_label in result:
            writer.writerow([img_name, predicted_label])

    print(f'预测结果已保存到 {csv_filename}')
