#!/bin/bash

sbatch --error=./runs/tf_dmrg_expt1_error.txt run_tf_dmrg_expt.sh 10 20 100 0.01 0.0001 runs/tf_dmrg_expt1_output.txt
sbatch --error=./runs/tf_dmrg_expt2_error.txt run_tf_dmrg_expt.sh 10 20 1000 0.01 0.0001 runs/tf_dmrg_expt2_output.txt
sbatch --error=./runs/tf_dmrg_expt3_error.txt run_tf_dmrg_expt.sh 10 200 1000 0.01 0.0001 runs/tf_dmrg_expt3_output.txt
sbatch --error=./runs/tf_dmrg_expt4_error.txt run_tf_dmrg_expt.sh 20 20 1000 0.1 0.0001 runs/tf_dmrg_expt4_output.txt
sbatch --error=./runs/tf_dmrg_expt5_error.txt run_tf_dmrg_expt.sh 20 40 1000 0.1 0.0001 runs/tf_dmrg_expt5_output.txt
sbatch --error=./runs/tf_dmrg_expt6_error.txt run_tf_dmrg_expt.sh 20 20 100 0.0001 0.0001 runs/tf_dmrg_expt6_output.txt
#sbatch --error=./runs/tf_dmrg_expt7_error.txt run_tf_dmrg_expt.sh 10 20 100 0.001 0.1 runs/tf_dmrg_expt7_output.txt
sbatch --error=./runs/tf_dmrg_expt8_error.txt run_tf_dmrg_expt.sh 10 20 100 0.001 0.1 runs/tf_dmrg_expt8_output.txt
sbatch --error=./runs/tf_dmrg_expt9_error.txt run_tf_dmrg_expt.sh 10 20 100 0.0001 0.1 runs/tf_dmrg_expt9_output.txt
sbatch --error=./runs/tf_dmrg_expt10_error.txt run_tf_dmrg_expt.sh 10 20 100 0.00001 0.1 runs/tf_dmrg_expt10_output.txt
