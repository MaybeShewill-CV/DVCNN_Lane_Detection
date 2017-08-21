#!/usr/bin/env bash
echo 'Start training DVCNN model'
python Tools/train_dvcnn.py --lane_line_dir /home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE/lane_line \
--non_lane_line_dir /home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE/non_lane_line \
--model_path /home/baidu/Road_Center_Line/DVCNN/model_def/DVCNN.json
echo 'Training procession complete'