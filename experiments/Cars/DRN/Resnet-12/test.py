import sys
import os
import torch
import yaml
sys.path.append('F:\VsCode-space\SRM-main-master')
from models.DRN import DRN
from utils import util
from trainers.eval import meta_test
import argparse

def test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disturb_num", help="channel number", type=int, default=1)
    parser.add_argument("--short_cut_weight", help="short cut weight", type=float, default=0.3)

    args = parser.parse_args()

    return args

with open('F:\VsCode-space\SRM-main-master\config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])

test_path = os.path.join(data_path,'cars/test_pre')
model_path = './model_ResNet-12.pth'
#model_path = '../../../../trained_model_weights/CUB_fewshot_cropped/FRN/ResNet-12/model.pth'

gpu = 0
torch.cuda.set_device(gpu)
args = test_parser()
model = DRN(resnet=True,args=args)
model.cuda()
model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)
model.eval()

with torch.no_grad():
    way = 5
    for shot in [1,5]:
        mean,interval = meta_test(data_path=test_path,
                                model=model,
                                way=way,
                                shot=shot,
                                pre=True,
                                transform_type=None,
                                trial=2000)
        print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))