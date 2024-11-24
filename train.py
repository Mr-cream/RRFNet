# coding: UTF-8
# @File   : train.py.py
# @Author : xingxg
# @Date   : 2024/11/24 21:23

from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO("./datasets/RRFNet.yaml")

    model.train(data="datasets/00_brain_tumor_3class.yaml", epochs=3)


