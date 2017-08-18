#!/usr/bin/env bash
echo 'Start training DVCNN model'
python Tools/train_dvcnn.py --lane_line_dir /home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE/lane_line \
--non_lane_line_dir /home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE/non_lane_line
echo 'Training procession complete'