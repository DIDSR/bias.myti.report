# Instructions on how to run the direct and indirect experiments on openHPC and Betsy
### Virtual environments
- OpenHPC
- Betsy 
```
source /scratch/ravi.samala/anaconda3/envs/venv_python38_2022116/bin/activate
```

## 1. Indirect experiments
### 1.a. 
- Shell script, parameter options
  - Generate data partitions
  Firstly generate one batch with 5 random seeds using [run_generate_partitions_v2.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_generate_partitions_v2.sh).
  In the script you will have the options to specify the output directory and name of generated data partitions.
  ```
  sh run_generate_partitions_v2.sh
  ```
    - Then run the script to limit 1 image per patient for generated data partitions using [run_csv_limit_images.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/bash_scripts/run_csv_limit_images.sh), including training, validation, validation_2 and independent_test datasets.
  ```
  sh run_csv_limit_images.sh
  ```
 - Run training
   - For baseline model training, using the script [run_finetune.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/run_finetune.sh)
  
   - For 2-stage model training, using the scripy [RSNA_run_2_step_train.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_2_step_train.sh)
 - Run deploy
  After training is done, we could deploy the model on validation_2 dataset by running []
- Plot/summarize results
  - After deployment is done, the 1st step is to ensemble results from 10 random states using [RSNA_run_ensemble.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_ensemble.sh)

  - The next step is to calculate bias measurements by running [RSNA_run_subgroup_analysis.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_subgroup_analysis.sh)

  - Then we could summarize all the results into one single csv file by [RSNA_run_summarize.sh](https://github.com/ravisamala/continual_learning_evaluation/blob/main/chexpert-model/betsy_scripts/RSNA_run_summarize.sh)
## 2. Direct experiments
