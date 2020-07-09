#!/bin/bash

sbatch --error=./runs/dense_fruits_expt1_error.txt run_dense_fruits_expt.sh 0.0001 100 100 ./runs/dense_fruits_expt1_output.txt
sbatch --error=./runs/dense_fruits_expt2_error.txt run_dense_fruits_expt.sh 0.001 100 100 ./runs/dense_fruits_expt2_output.txt
sbatch --error=./runs/dense_fruits_expt3_error.txt run_dense_fruits_expt.sh 0.0001 1000 100 ./runs/dense_fruits_expt3_output.txt
sbatch --error=./runs/dense_fruits_expt4_error.txt run_dense_fruits_expt.sh 0.0001 100 1000 ./runs/dense_fruits_expt4_output.txt
sbatch --error=./runs/dense_fruits_expt5_error.txt run_dense_fruits_expt.sh 0.0005 100 100 ./runs/dense_fruits_expt5_output.txt

sbatch --error=./runs/dense_fruits_expt6_error.txt run_dense_fruits_expt.sh 0.0001 5000 100 ./runs/dense_fruits_expt6_output.txt
