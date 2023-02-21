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


# INPUT_CSV -> current test/validation/etc. csv file to limit the number of images/patient
INPUT_CSV=""

# OUTPUT_CSV -> filepath + filename to save output csv to
    # NOTE: WILL overwrite files if provided a filepath to an existing file!
OUTPUT_CSV=""

# NUM_IMAGES -> maximum number of images for each patient
NUM_IMAGES=1

python ../src/csv_limit_images.py -i ${INPUT_CSV} \
                                  -o ${OUTPUT_CSV} \
                                  -n ${NUM_IMAGES} 