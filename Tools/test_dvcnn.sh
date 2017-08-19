#!/usr/bin/env bash
echo 'Start testing DVCNN model'
python Tools/test_dvcnn.py --model_path DVCNN/model_def/DVCNN.json --weights_path DVCNN/model/dvcnn.ckpt-2999 \
--lane_line_dir /home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE/Validation/lane_line \
--non_lane_line_dir /home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE/Validation/non_lane_line
echo 'Testing procession complete'