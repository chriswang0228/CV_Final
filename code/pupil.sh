#!/bin/bash
#PBS -l select=1:ncpus=4:gpu_id=6
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N pupil_DLV3
cd /home/chriswang/class/cv_EE/final/code								

source ~/.bashrc											
conda activate cv2				

module load cuda-11.7										
python3 submission.py
