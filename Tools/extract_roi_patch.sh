#!/usr/bin/env bash
echo 'Start extracting roi patches ...'
python Tools/extract_roi_patch.py \
--top_image_dir /home/baidu/DataBase/Road_Center_Line_DataBase/Origin/Train/top_view_crop \
--fv_image_dir /home/baidu/DataBase/Road_Center_Line_DataBase/Origin/Train/fv_view \
--top_rois_dir /home/baidu/DataBase/Road_Center_Line_DataBase/Extract_Roi/top_view \
--fv_rois_dir /home/baidu/DataBase/Road_Center_Line_DataBase/Extract_Roi/front_view
echo 'Extracting procession complete'