#!/bin/bash
# # Program to 
# # to run: >> ./run_generate_partitions.sh
# # 
# # 
# #
EXE="../src/generate_partitions.py"
# declare -a IN_summary=("/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__MIDRC_RICORD_1C.json" \
# 						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__COVID_19_AR.json" \
#						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__COVID_19_NY_SBU.json" \
#						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__open_AI.json" \
#						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__open_RI.json")
# General options
# declare -a IN_summary=("/gpfs_projects/ravi.samala/OUT/2022_CXR/data_summarization/20221010/20221010_summary_table__open_A1.json")

declare -a IN_summary=("/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/summary_table__MIDRC_RICORD_1C.json")
# 						"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/summary_table__COVID_19_NY_SBU.json")
# OUT_dir="/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3"
# OUT_dir="/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v4"
OUT_dir="/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs"
RAND_SEED=2050
# split/step options
N_steps=1
ACCUMULATE=0 # [0,1], portion of previous step to include 
PERCENT_TEST_PARTITION=0.2
ADD_JOINT_VAL=False
SPLIT_TYPE=equal
# SPLIT_TYPE="0.25,0.125,0.125,0.125,0.125,0.125,0.125" # Does not include the validation step, which is separated before the split	
						# # equal: N_steps equal sized splits
				   		# # increasing: each step is larger than the last
						# # random: random step sizes (adding up to 100% of input)
						# # custom (ex. 0.2,0.8): set the split sizes, asumes that each size is separated by just a comma, 
							# # that the number of sizes is equal to N_steps, and that the values add to 1

# control # images per patient (NOTE: will not apply to final step, which it assumes is a validation split)
OPTION=0	# # 0: all images/patient
			# # 1: first min N number of images/patient
MIN_N_IMAGES_PATIENT=1	# # min number of patients per image, ordered by time

STRATIFY=sex	# TO see the exact groups, see subgroup_dict in generate_partitions.py
# STRATIFY=False
# TASKS="F,M,DC,DX"
TASKS="F,M,CR,DX,Yes,No,Black_or_African_American,White" # see supported values in subgroup_dict in generate_partitions.py
															# # Separate values with commas, if a value has spaces, replace with underscore
# STEP_repos="0-2__MIDRC_RICORD_1C/3,4__COVID_19_NY_SBU" ## WIP
# PARTITION_NAME='custom_split_7_steps_no_accumulation'
PARTITION_NAME="time_comparison_MIDRC_RICORD_1C"

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

	echo "======= RAND ${RAND} ======="
	# #
	# OUT_dir2="${OUT_dir}/RAND_${RAND}_OPTION_${OPTION}_custom_acc_${N_steps}_steps"
	OUT_dir3="${OUT_dir}/${PARTITION_NAME}"
	mkdir $OUT_dir3
	OUT_dir2="${OUT_dir}/${PARTITION_NAME}/RAND_${RAND}"
	mkdir $OUT_dir2
	python $EXE $param_IN_summary -o $OUT_dir2 \
								  -r $RAND  \
								  -s $OPTION \
								  -m $MIN_N_IMAGES_PATIENT \
								  -steps $N_steps \
								  -split_type $SPLIT_TYPE \
								  -accumulate $ACCUMULATE \
								  -stratify $STRATIFY \
								  -tasks $TASKS \
								  -p $PERCENT_TEST_PARTITION \
								  -add_joint_validation $ADD_JOINT_VAL \
								  -partition_name $PARTITION_NAME 
done
