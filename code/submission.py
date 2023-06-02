import os
import numpy as np
import cv2
from tqdm import tqdm
from utils import AverageMeter
import torch
from torchvision import transforms as T
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('../model_weight/DLV3.pt').eval().to(device)
exp_name = '4'
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

def my_awesome_algorithm(image):
    image = t(image).to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        result=torch.argmax(output, dim=1).cpu().squeeze(0).numpy()
        if result.sum()<400:
            conf = 0.0
        else:
            conf = 1.0
    return result, conf
    

def benchmark(dataset_path: str, subjects: list, times:str):
    """Compute the weighted IoU and average true negative rate
    Args:
        dataset_path: the dataset path
        subjects: a list of subject names

    Returns: benchmark score

    """
    os.mkdir(f'../submission/{times}')
    os.mkdir(f'../submission/{times}/solution')
    output_conf = []
    sequence_idx = 0
    for subject in subjects:
        os.mkdir(f'../submission/{times}/solution/{subject}')
        length = len(os.listdir(f'../dataset/{subject}/{subject}/'))
        for action_number in range(length):
            os.mkdir(f'../submission/{times}/solution/{subject}/{action_number+1:02d}')
            image_folder = os.path.join(dataset_path, subject, subject, f'{action_number + 1:02d}')
            sequence_idx += 1
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            output_conf = []
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                image_name = os.path.join(image_folder, f'{idx}.jpg')
                image = cv2.imread(image_name)
                output, conf = my_awesome_algorithm(image)
                output_conf.append(conf)
                cv2.imwrite(f'../submission/{times}/solution/{subject}/{action_number+1:02d}/{idx}.png', output)
            np.savetxt(f'../submission/{times}/solution/{subject}/{action_number+1:02d}/conf.txt', output_conf, fmt="%d", delimiter="\r\n")

if __name__ == '__main__':
    dataset_path = '../dataset/'
    subjects = ['S5', 'S6', 'S7', 'S8']
    benchmark(dataset_path, subjects, exp_name)
