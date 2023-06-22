# Bash Script Templates
This folder contains a number of template bash script files for different common ways of using the generate partitions code. Feel free to add your own templates, but please do not edit the existing ones (unless necessary to fix bugs). If you add a template, please add a brief description [below](#templates). If a template is created for a specific experiment, I recommend noting the experiment in the description.

---
## How to use
It is recommended to make copies of the template files before use, that way you can make any necessary modifications while maintaining the template.
- fill in the missing information (typically `save_dir` and `experiment_name`) 
- adjust the number(s) for RANDs and batches as desired
- modify `repo` as needed; may include multiple
### To use on Betsy:
- modify `EXE` to include the entire file path
- add the `--betsy` argument
- add the following line to activate the current virtual environment:

source /scratch/ravi.samala/venvs/conda_venv/venv_python310_20230608/bin/activate

---
## Templates

### [Basic Partitioning](basic_partitioning.sh)
Generates train (50%), validation-1 (10%), validation-2 (20%) and test (20%) partitions for a single step. Randomly samples the existing distribution.

### [Basic Batching](basic_batching.sh)
Similar to [basic partitioning](#basic-partitioningbasicpartitioningsh), however the validation-2 and test partitions are batched.
### [Equal Stratification by Patient Sex, Race, and COVID status](equal_strat_sex_race_COVID.sh)
Similar to [basic partitioning](#basic-partitioningbasicpartitioningsh), except that the partitions are equally stratified into the subgroups: Female-Black-Negative, Female-Black-Positive, Female-White-Negative, Female-White-Positive, Male-Black-Negative, Male-Black-Positive, Male-White-Negative, Male-White-Positive,.



