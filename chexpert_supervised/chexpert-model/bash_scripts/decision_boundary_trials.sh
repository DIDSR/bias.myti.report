#!/bin/bash
# 
# #
# If the program erros and crashed, running it again should resume where it left off
# #
# For SPIE2023 runs:
# change RAND # below
# will automatically select accompanying training csv as input,
# and save in the folder atm2/decision_boundaries/RAND_{RAND#}
# #
# to change sample # and/or subgroup combinations, look at db_trials trial_setup
# #
# the csv for each subgroup (listing percents of FCR, FDX, MCR, MDX for each plot) is updated as each plot is created.
# the overall summary csv is only updated after all 250 samples of the subgroup are created, but a checkpoint update
# of the current mean and std is printed every 10 plots made
# 
RAND=0
echo Beginning Trial
python db_trials.py --dataset custom \
                    --together True \
                    --ckpt_path /gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/atm2/RAND_${RAND}/full_MIDRC_RICORD_1C/best.pth.tar \
                    --phase valid \
                    --moco False \
                    --inference_only \
                    --gpu_ids 3 \
                    
                    
