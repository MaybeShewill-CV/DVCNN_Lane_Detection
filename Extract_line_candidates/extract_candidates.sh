#!/usr/bin/env bash
echo "Start extracting lane line proposal from top view image"
python extract_candidate.py --top_file_dir data/top --extract_save_dir candidates_result/top
echo "Process done"
