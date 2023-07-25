# Instructions on how to run the direct and indirect experiments on openHPC and Betsy
### Virtual environments
- OpenHPC
```
source /gpfs_projects/ravi.samala/venvs/venv_Py310/bin/activate
```
- Betsy 
```
source /scratch/ravi.samala/anaconda3/envs/venv_python38_2022116/bin/activate
```

## 1. Indirect experiments
### 1.a. 
- Data preparation  
    - Firstly generate 1 batch with 5 random seeds. For both **OpenHPC** and **Betsy** use [run_generate_partitions_v2.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_generate_partitions_v2.sh).
  ```
  sh run_generate_partitions_v2.sh
  ```
  In the script the user will have the options to specify the output directory and name of generated data partitions on line 43 and 44.
  ```
  PARTITION_NAME=batch_0
  OUT_dir="/scratch/yuhang.zhang/OUT/temp/"
  ```
    - Then for both **OpenHPC** and **Betsy**, run the script to limit 1 image per patient for generated data partitions using [run_csv_limit_images.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_csv_limit_images.sh), including training, validation, validation_2 and independent_test datasets.
  ```
  sh run_csv_limit_images.sh
  ```
  Again the user should specify the main directory that contains the data csv files on line 13.
  ```
  MAIN_PATH="/scratch/yuhang.zhang/OUT/temp/batch_0/"
  ```
 - Run training
   - For baseline model training, using the script [run_finetune.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/run_finetune.sh).  
   On line 13 the user should specify the main directory that contains csv data files.
    ```
    MAIN_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}
    ```   
   - For 2-stage model training, using the script [RSNA_run_2_step_train.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_2_step_train.sh).  
   Again, the main directory should be assigned on line 20.
    ```
    MAIN_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}
    ```
   Noted that for running both scripts on **OpenHPC**, the user should comment out the virtual environment activation commend at the top, which is used for sbatch mode on **Betsy**.
 - Run deploy  
   After training is done, the user could deploy the model on validation_2 dataset by running [run_validation_2.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/run_validation_2.sh). The main directory should be specified on line 10.  
    Again noted that for running on **OpenHPC**, the user should comment out the virtual environment activation commend at the top, which is used for sbatch mode on **Betsy**.
  
- Plot/summarize results
  - After deployment is done, the 1st step is to ensemble results from 10 random states using [RSNA_run_ensemble.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_ensemble.sh)  
  The main directory is specified on line 22.

  - The next step is to calculate bias measurements by running [RSNA_run_subgroup_analysis.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_subgroup_analysis.sh)  
  The main directory is specified on line 21.  
  The user can specify which subgroups to calculate bias measurements (e.g., sex, race)

  - Then the user could summarize all the results into one single csv file by [RSNA_run_summarize.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_summarize.sh)  
  The main directory is specified on line 11.
### 1.b.
- Running 1.b. is pretty much the same with running 1.a., except for several changes listed below.
  - When running 2-step model training using [RSNA_run_2_step_train.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_2_step_train.sh), the user has to change the tasks to classify "race" instead of "sex" for the stage-1 model on line 13 and 14.
  ```
  TASK_1=Black
  TASK_1_R=White
  ```
  - When calculating bias measurements using [RSNA_run_subgroup_analysis.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_subgroup_analysis.sh), the user should modify "TEST_SUBGROUP" to "race" on line 14.
  ```
  TEST_SUBGROUP="race"
  ```
### 1.c.
- Running 1.c. is pretty much similar to 1.a., except for several changes listed below.
  - For data preparation, after running [run_csv_limit_images.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_csv_limit_images.sh), the user should run [run_csv_data_split.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/betsy_scripts/run_csv_data_split.sh) to split the training/validation data. On line 7, 8 and 9, the user can specify the input and output filenames for the csv data file.
``` 
INPUT_FILE="train.csv"
OUTOUT_1="train_1_baseline.csv"
OUTOUT_2="train_2_baseline.csv"
```
Line 14 and 15 are the places where user can specify the input and output directory for these data files.
  - For training the 2-stage models using [RSNA_run_2_step_train.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_2_step_train.sh), user should assign seperate training/validation dataset for stage 1 and 2.  
  Stage 1: line 25, 53 and 26, 54 change to "train_1_baseline.csv" and "validation_1_baseline.csv" in our current example.  
  Stage 2: line 86, 115 and 87, 116 change to "train_2_baseline.csv" and "validation_2_baseline.csv".
  - For baseline model using [run_finetune.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/run_finetune.sh), the same training/validation dataset from stage 2 should be assigned on line 20 and 21 respectively.
## 2. Direct experiments
### 2.a. 
- Data preparation
  - First 2 steps are the same as 1.a, the user can run [run_generate_partitions_v2.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_generate_partitions_v2.sh) and [run_csv_limit_images.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_csv_limit_images.sh) to generate training, validation, validation_2 and independent_test datasets.
  - After that, the user should run [run_csv_data_split.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/betsy_scripts/run_csv_data_split.sh) to construct training/validation datasets for experimental (biased) models. To achieve that, the user should set the fraction to 0 on line 10, and name the output files properly on line 8 and 9.
``` 
OUTOUT_1="train_50FP.csv"
OUTOUT_2="train_50FP.csv"
FRACTION=0.5
``` 
  Then run the same script [run_csv_data_split.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/betsy_scripts/run_csv_data_split.sh) to construct evenly splitted training/validation datasets for baseline models. In this step, the user should set the fraction to 0.5 on line 10.
- Run training
  - Baseline models should use [run_finetune.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/run_finetune.sh) with training/validation datasets generated for baseline models (assigning on line 20 and 21).
  - Similarly, experimental (biased) models should also use [run_finetune.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/run_finetune.sh) with training/validation datasets generated for experimental (biased) models (assigning on line 20 and 21).
### 2.a. (Additional Levels of Control)
- Data preparation
  - First 2 steps are exactly the same as 2.a., including running [run_generate_partitions_v2.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_generate_partitions_v2.sh) and [run_csv_limit_images.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_csv_limit_images.sh) to generate training, validation, validation_2 and independent_test datasets.
  - After that, the user should run [run_csv_data_split.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/betsy_scripts/run_csv_data_split.sh) to construct a series of training/validation dataset. The user should set the fraction to 0, 0.1, 0.25, 0.5 on line 10 respectively. The output csv files should be properly name on line 8 and 9. Noted that one input fraction can output two seperate csv files (e.g., a fraction of 0.1 can generate partitions of 10% female-positive, as well as 90% female-positive).
- Run training
  - Training steps are pretty much the same as 2.a. The user should properly assign training/validation dataset used for the experiment on line 20 and 21 using [run_finetune.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/run_finetune.sh). 
### 2.b. 
- Both data preparation and model training for 2.b. is pretty much the same as 2.a. The user only need to change line 11 in [run_csv_data_split.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/betsy_scripts/run_csv_data_split.sh) to "race" during data preparation.
``` 
TEST_SUB="race"
``` 
### 2.b. (Additional Levels of Control)
- Data preparation
  - Again, first 2 steps are exactly the same as 2.b by running [run_generate_partitions_v2.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_generate_partitions_v2.sh) and [run_csv_limit_images.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_csv_limit_images.sh). After that, the user should run [run_csv_data_split.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/betsy_scripts/run_csv_data_split.sh) similar to 2.a. (Additional Levels of Control) while change TEST_SUB to "race" on line 11.
- Run training
  - Training steps are the same as 2.b. The user should properly assign training/validation dataset used for the experiment on line 20 and 21 using [run_finetune.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/run_finetune.sh) to properly set up the series of experiments. 
