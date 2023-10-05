#!/bin/bash
#Limit the number of images/patient in an existing train/validation/etc. csv by selecting the images taken closest
#to a relevant test (positive test for positive patients, negative test for negative patients)
# NOTES:
    # Currently only implemented for open-A1
    # Automatically removes patients that do not have information for days from study to relevant test
        # to include patients/images without test information, pass the argument --allow_null
            # This will still limit the number of images per patient, however for patients with studies missing
            # test date information, the images chosen will be selected randomly!
    # output csv includes a new column "days_from_study_to_test" a negative value indicates that the test was taken before
    # the image, while a positive value indicates that the test was taken after the image
    # when a single patient has multiple images taken on the same day, the image selection is random from those images
MAIN_PATH="/scratch/yuhang.zhang/OUT/temp/batch_0/"
# NUM_IMAGES -> maximum number of images for each patient
NUM_IMAGES=1
# INPUT_CSV -> current test/validation/etc. csv file to limit the number of images/patient
INPUT_CSV_VA2=$MAIN_PATH"/validation_2.csv"
INPUT_CSV_TS=$MAIN_PATH"/independent_test.csv"
OUTPUT_CSV_VA2=$MAIN_PATH"/validation_1_image.csv"
OUTPUT_CSV_TS=$MAIN_PATH"/test_1_image.csv"
python ../src/csv_limit_images.py -i ${INPUT_CSV_VA2} \
                                  -o ${OUTPUT_CSV_VA2} \
                                  -n ${NUM_IMAGES} \
                                  --allow_null
python ../src/csv_limit_images.py -i ${INPUT_CSV_TS} \
                                  -o ${OUTPUT_CSV_TS} \
                                  -n ${NUM_IMAGES} \
                                  --allow_null
for RAND in 0 1 2 3 4
do
INPUT_CSV_VA=$MAIN_PATH"/RAND_${RAND}/validation.csv"
INPUT_CSV_TR=$MAIN_PATH"/RAND_${RAND}/train.csv"
# OUTPUT_CSV -> filepath + filename to save output csv to
    # NOTE: WILL overwrite files if provided a filepath to an existing file!
OUTPUT_CSV_VA=$MAIN_PATH"/RAND_${RAND}/validation.csv"
OUTPUT_CSV_TR=$MAIN_PATH"/RAND_${RAND}/train.csv"

python ../src/csv_limit_images.py -i ${INPUT_CSV_TR} \
                                  -o ${OUTPUT_CSV_TR} \
                                  -n ${NUM_IMAGES} \
                                  --allow_null
python ../src/csv_limit_images.py -i ${INPUT_CSV_VA} \
                                  -o ${OUTPUT_CSV_VA} \
                                  -n ${NUM_IMAGES} \
                                  --allow_null
done