# -*- coding: utf-8 -*-

import torch
import argparse
from collections import OrderedDict

def change_model(args):
    src_model = torch.load(args.src_path, map_location='cpu')
    all_name = []
    for name, v in src_model["state_dict"].items():
        if name.startswith("student."):
            all_name.append((name[8:], v))
        elif name.startswith("student_"):
            continue
        elif name.startswith("teacher.") or name.startswith("teacher_") or "fgd_" in name:
            continue
        else:
            all_name.append((name, v))
            
    state_dict = OrderedDict(all_name)
    src_model['state_dict'] = state_dict
    if 'optimizer' in src_model:
        src_model.pop('optimizer')
    torch.save(src_model, args.dst_path)

           
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer CKPT')
    parser.add_argument('--src_path', type=str, default='work_dirs/fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco/epoch_24.pth', 
                        metavar='N',help='src_model path')
    parser.add_argument('--dst_path', type=str, default='retina_res50_new.pth',metavar='N', 
                        help = 'pair path')
    args = parser.parse_args()
    change_model(args)