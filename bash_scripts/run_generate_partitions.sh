#!/bin/bash
# # Program to 
# # to run: >> ./run_generate_partitions.sh
# # 
# # 
# #
EXE="../generate_partitions.py"
# declare -a IN_summary=("/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__MIDRC_RICORD_1C.json" \
# 						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__COVID_19_AR.json" \
#						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__COVID_19_NY_SBU.json" \
#						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__open_AI.json" \
#						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__open_RI.json")
# General options
declare -a IN_summary=("/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/summary_table__MIDRC_RICORD_1C.json")
OUT_dir="/gpfs_projects/alexis.burgon/OUT/2022_CXR/temp"
RAND_SEED=2050
# split/step options
N_steps=2
ACCUMULATE=False	#NOTE: will not accumulate for the last step, which it assumes is a validation split
SPLIT_TYPE=0.2,0.8 		# # equal: N_steps equal sized splits
				   		# # increasing: each step is larger than the last
						# # random: random step sizes (adding up to 100% of input)
						# # custom (ex. 0.2,0.8): set the split sizes, asumes that each size is separated by just a comma, 
							# # that the number of sizes is equal to N_steps, and that the values add to 1

# control # images per patient (NOTE: will not apply to final step, which it assumes is a validation split)
OPTION=1	# # 0: all images/patient
			# # 1: first min N number of images/patient
MIN_N_IMAGES_PATIENT=1	# # min number of patients per image, ordered by time

STRATIFY=True

STRAT_GROUPS=F-DX,F-CR,M-CR,M-DX


# #
# #
# #
# # -----------------------------------------------------------------------
# # -----------------------------------------------------------------------
# # -----------------------------------------------------------------------
# #
# for RAND in 0 1 2 3 4 5 6 7 8 9
for RAND in 0
do
	# # get length of an array
	arraylength=${#IN_summary[@]}
	# #
	param_IN_summary=""
	# # create input params
	for (( i=0; i<${arraylength}; i++ ));
	do
		param_IN_summary="${param_IN_summary} -i ${IN_summary[$i]}"
	done
	# #
	OUT_dir2="${OUT_dir}/RAND_${RAND}"
	mkdir $OUT_dir2
	python $EXE $param_IN_summary -o $OUT_dir2 \
								  -r $RAND  \
								  -s $OPTION \
								  -m $MIN_N_IMAGES_PATIENT \
								  -steps $N_steps \
								  -split_type $SPLIT_TYPE \
								  -accumulate $ACCUMULATE \
								  -stratify $STRATIFY \
								  -strat_groups $STRAT_GROUPS
done
