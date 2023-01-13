import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='C:/tujianfeng/Bayesian-Crowd-Counting-master/teedy/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='C:/tujianfeng/Bayesian-Crowd-Counting-master/teedy/vgg/0106-190713',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


def density(inputs):
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    epoch_minus = []

    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            out_str = 'name:{}, 密度估计：{}, 真实人数：{}, 估计人数：{}, 误差：{}'.format(name, outputs, count[0].item(),torch.sum(outputs).item(),temp_minu)
            print(out_str)
            epoch_minus.append(temp_minu)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)

    return name, outputs, count[0].item(),torch.sum(outputs).item(),temp_minu

if __name__ == '__main__':
    args = parse_args()
    density(args)





#python test.py --data_dir <directory of processed data> --save_dir <directory of log and model>