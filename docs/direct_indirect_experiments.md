# Instructions on how to run the direct and indirect experiments on openHPC and Betsy
### Virtual environments
- OpenHPC
- Betsy 
```
source /scratch/ravi.samala/anaconda3/envs/venv_python369/bin/activate
```

## 1. Indirect experiments
### 1.a. 
- Data preparation
  - Generate data partitions
  Firstly generate 1 batch with 5 random seeds using [run_generate_partitions_v2.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_generate_partitions_v2.sh).
  ```
  sh run_generate_partitions_v2.sh
  ```
  In the script the user will have the options to specify the output directory and name of generated data partitions on line 43 and 44.
  ```
  PARTITION_NAME=batch_0
  OUT_dir="/scratch/yuhang.zhang/OUT/temp/"
  ```
    - Then run the script to limit 1 image per patient for generated data partitions using [run_csv_limit_images.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_csv_limit_images.sh), including training, validation, validation_2 and independent_test datasets.
  ```
  sh run_csv_limit_images.sh
  ```
  Again the user should specify the main directory that contains the data csv files on line 13.
  ```
  MAIN_PATH="/scratch/yuhang.zhang/OUT/temp/batch_0/"
  ```
 - Run training
   - For baseline model training, using the script [run_finetune.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/run_finetune.sh)
   On line 13 the user should specify the main directory that contains csv data files.
    ```
    MAIN_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}
    ```   
   - For 2-stage model training, using the script [RSNA_run_2_step_train.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_2_step_train.sh)
   Again, the main directory should be assigned on line 20.
    ```
    MAIN_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}
    ```   
 - Run deploy  
    - After training is done, the user could deploy the model on validation_2 dataset by running [run_validation_2.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/run_validation_2.sh)  
    The main directory should be specified on line 10.
  
- Plot/summarize results
  - After deployment is done, the 1st step is to ensemble results from 10 random states using [RSNA_run_ensemble.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_ensemble.sh)  
  The main directory is specified on line 22.

  - The next step is to calculate bias measurements by running [RSNA_run_subgroup_analysis.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_subgroup_analysis.sh)  
  The main directory is specified on line 21.  
  The user can specify which subgroups to calculate bias measurements (e.g., sex, race)

  - Then the user could summarize all the results into one single csv file by [RSNA_run_summarize.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_summarize.sh)  
  The main directory is specified on line 11.
## 2. Direct experiments
