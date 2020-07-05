#!/bin/bash

sbatch --error=./runs/tn_fruits_expt1_error.txt run_tn_fruits_expt.sh 0.0001 100 100 ./runs/tn_fruits_expt1_output.txt
sbatch --error=./runs/tn_fruits_expt2_error.txt run_tn_fruits_expt.sh 0.001 100 100 ./runs/tn_fruits_expt2_output.txt
sbatch --error=./runs/tn_fruits_expt3_error.txt run_tn_fruits_expt.sh 0.0001 1000 100 ./runs/tn_fruits_expt3_output.txt
sbatch --error=./runs/tn_fruits_expt4_error.txt run_tn_fruits_expt.sh 0.0001 100 1000 ./runs/tn_fruits_expt4_output.txt
sbatch --error=./runs/tn_fruits_expt5_error.txt run_tn_fruits_expt.sh 0.0005 100 100 ./runs/tn_fruits_expt5_output.txt
