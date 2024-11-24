# coding: UTF-8
# @File   : val.py
# @Author : xingxg
# @Date   : 2024/4/4 16:28

from ultralytics import YOLO

if __name__ == '__main__':


    model = YOLO("./pretrained/RRFNet/RRFNet-n.pt")


    metrics = model.val(data="./datasets/00_brain_tumor_3class.yaml", batch=1)
