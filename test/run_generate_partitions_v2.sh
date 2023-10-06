#!/bin/bash
# ===================   NOTES   ===================
# Not Yet Implemented:
    # stratification by repository
# Not yet tested:
    # repositories other than open_A1
    # multiple repositories
# Program outputs:
    # train, test, and validation partitions
    # by-image and by-patient summary files, which have the number of patients and images, respectively,  within each subgroup and split
    # partition_info.log -> holds the input arguments, for future reference

# =================== SETTINGS ===================

declare -a IN_summary=("/scratch/alexis.burgon/2022_CXR/data_summarization/20221010/summary_table__open_A1.json")
# declare -a IN_summary=("/scratch/alexis.burgon/2022_CXR/data_summarization/20220823/summary_table__MIDRC_RICORD_1C.json")
# TASKS specifies the output columns in addition to patient_id and img file Path
    # Not necessarily indicative of finetuning tasks, although all finetuning tasks must be included 
declare -a TASKS=("M" "F" 'White' 'Black' 'Yes' 'No') 

TEST_SIZE=0.2 
VAL_SIZE=0.1
VAL_2_SIZE=0.2

REMAINING_TO_TEST=True
# if TEST_RAND=None, each RAND parition will have their own separate independent test file
    # if TEST_RAND is a number, then all RAND paritions will share an independent test file, generated with the random seed TEST_RAND
TEST_RAND=2
VAL_2_RAND=1
# For more advanced stratification settings, see the top of generate_partitions_v2.py
    # to utilize any kind of stratification in any of the test/train/validation partitions,
    # STRATIFY must equal True
STRATIFY=True

# ALLOW_OTHER specifies whether patients from subgroups other than those specified in group_dict (constant from generate_partitions_v2.py)
    # will be included in any of the output test/train/validation partitions. (Only applies to non-stratified splits)
    # ex. only White and Black are listed as races in group_dict, so patients who are not Black or White will be excluded if ALLOW_OTHER=False
ALLOW_OTHER=False

# Save options
    # folder w/ name {PARTITION_NAME} will automatically be created in {OUT_dir}, RAND folders will be created within
OUT_dir="/scratch/yuhang.zhang/OUT/temp/"

# Number of imgs/patient selection_modes = random, first, last (how to select images if a max is specified)
    # NOTE: random selection mode will potentially cause issues with TEST_RAND != None, use with caution
MIN_IMG=0
MAX_IMG=None
IMG_SELECTION=last
for BATCH in 0
do
PARTITION_NAME=batch_${BATCH}
for RAND in 0
do
# # ==================================================================
# # Nothing below this point should need to be edited for regular use
# # ==================================================================
# # in summary processing
	arraylength=${#IN_summary[@]}
	param_IN_summary=""
	for (( i=0; i<${arraylength}; i++ ));
	do
		param_IN_summary="${param_IN_summary} -i ${IN_summary[$i]}"
	done
# # tasks processing
    arraylength=${#TASKS[@]}
    param_TASKS=""
    for (( i=0; i<${arraylength}; i++));
    do
        param_TASKS="${param_TASKS} -tasks ${TASKS[$i]}"
    done
# # ======================================
echo ================ RAND ${RAND} ================
python ../src/generate_partitions_v2.py  $param_IN_summary \
                                         $param_TASKS \
                                         -random_seed $RAND \
                                         -random_seed_initial ${BATCH} \
                                         -test_size $TEST_SIZE \
                                         -validation_size_2 $VAL_2_SIZE \
                                         -consistent_test_random_state $TEST_RAND \
                                         -consistent_validation_2_random_state $VAL_2_RAND \
                                         -validation_size $VAL_SIZE \
                                         -partition_name $PARTITION_NAME \
                                         -save_dir $OUT_dir \
                                         -allow_other $ALLOW_OTHER \
                                         -stratify $STRATIFY \
                                         -min_img_per_patient $MIN_IMG \
                                         -max_img_per_patient $MAX_IMG \
                                         -patient_img_selection_mode $IMG_SELECTION \
                                         -remaining_to_test $REMAINING_TO_TEST
done
done