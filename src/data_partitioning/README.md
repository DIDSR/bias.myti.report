# Generate Partitions
This module separates the designated data sets into steps and partitions of designated sizes and subgroup distributions. This file gives a brief overview of different ways to use this module, as well as the different [input arguments](argument_parser.py). See [the bottom of this document](#resources) if you have questions.

_This module deliberately works independently of the other files in the repository, so that it may be applied beyond the repository's scope. However, the output files (for CXR) are specifically formatted to work for this repository. Due to this module's independence, all previous generate_partitions files should continue to work as they did before._

---
**Current Status**:
- documentation is a WIP, however most input arguments are described [below](#arguments).
- currently works for segmentation as well as CXR (pass `--repo BraTS`), however the output csv format will likely need to be changed in the near future
- untested on betsy, and certain features that require the use of additional files may not be implemented, or not implemented completly (including using PCR test date, portable/non-portble attributes, and certain data repositories)
---

## Using this module
This module can be used from the command line or through bash scripts; see the [templates folder](bash_script_templates/) for bash script templates / examples of how to pass input arguments. 

### Outputs
Each designated partition will have an output csv file in the appropriate "RAND" or "batch" folder. Each "RAND" folder will also contain a file named `partition_settings.log`, which contains the information necessary to recreate the generated partition, including all command line arguments passed.

An additional folder named partition_summary will be created within each "RAND" folder. This partition_summary folder will contain:
1. a file named "Partition_summary.csv", which includes the number of patients and images in each subgroup/partition
2. A distribution_summary image for each attribute specified in `--attributes`, `--tasks` or `--summary-attributes`, showing the distribution of that attribute for each partition
3. an image file called "partition_overlap_all" which shows the overlap of patients between different partitions. _Unless generating multiple steps, there should never be any overlap between partitions._ If generating multiple steps, there will be additional "partition_overlap" files, one for each partition, to be used to ensure that any accumulation or replace arguments worked aas intended.
---
## Arguments
**Required arguments are bolded**



### General Arguments
- **`---r/-random-seed`** : random state
- **`--repository/--repo`** : source repositoriy (or repositories)
- **`--save-location/--save-loc`** : output directory
- `--tasks` : designates which attributes wil be converted to task format in the output files. Ex. `--tasks COVID_positive sex` will result in the output file having 4 binary columns: Yes, No, F, M
- `--experiment-name` : if provided, a direcotry will be created in `--save-location`, and the batch/RAND folders will be created within; if not provided, batch/RAND folders will be created in `--save-location`
- `--overwrite` : removes a check to see if the RAND folder already exists. _Note: be cautious using this argument with batches, a safer alternative is to manually delete the RAND folder._
- `--id-column` : default behavior stratifies by 'patient_id', to stratify by image, pass `--id-column Path`. _Note: stratification by image is currently untested._
- `--betsy` : pass if running on betsy 
### Attribute Arguments
To see a list of available attributes, see [constants.py](constants.py); for CXR, "repo" is also a supported attribute.
Currently, "portable" is an accepted attribute for _summary purposes only_, as some patients have both portbale and non-portable studies, it does not currently work for stratification _unless only 1 image per patient is used_.
- `--attributes` : the attributes used to determine patient subgroup and considered during stratification.
- `--summary-attributes` : attributes that are _not_ considered when determining patient subgroup, but are used when generating the output summary files.
- `--subtract-from-smallest-subgroup` : the number of patients to be subtracted from the smallest subgroup to allow for increased variability between RANDs.
- `--allow-not-reported` : pass to allow the use of samples that are missing information relating to 1+ of the attributes designated
- `--exclude-attributes` : specific attribute values to remove from all partitions (ex. to remove DX images `--exclude-attributes modality:DX`); can be used for multiple values of one attribute using a comma (ex. `--exclude-attributes sex:F,M`)
### Partition Arguments
- **`--partitions`** : names of partitions to generate
- **`--partition-sizes`** : relative size of each partition
- `--partition-distributions` : distribution of each partition; see [subgroup_distribution_options](subgroup_distribution_options.py) for distribution information.
- `--accumulate`: only applicable with multiple steps; whether to accumulate samples in this partition between steps or not
- `--replace`: only applicable with multiple steps; whether to maintain a constant partition size by using samples from the previous step or not
  - `--replace-by-subgroup` : if passed, will maintain the subgroup distribution
- `--batch` : list of partitions to batch
  - `--batch-rand` : the random state used for the partitions listed in `--batch`

#### How to use:
Partition arguments follow the that the partitions are provided in `--partitions`; for example, to generate a train and a test partition, where the train partition is twice the size of the train partition: `--partitions train test --partition-sizes 2 1`

For most arguments (with the exception of `--partition sizes`), passing a single argument will apply it to all partitions. For example, if you have train, test, and validation partitions, and want to have and equal subgroup distribution in all of them: `--partitions train validation test --partition-distributions equal`
### Step Arguments
In general, step arguments work similarly to partition arguments.
- `--steps`: number of steps (`int`)
- `--step-sizes` : relative size of each step
- `--step-distibutions` : distribution of each step; see [subgroup_distribution_options](subgroup_distribution_options.py) for distribution information.
### Limiting the number of images per patient
- `--images-per-patient` : number of images per patient
- `--image-limit-type` : how the limit is applied
- `--image-limit-method` : how the images are selected
  - `--remove-null-PCR` : pass if using the PCR limit method to remove samples missing PCR test information

---
## Resources
### Terminology
- **batches/RANDs**: To allow for 1+ partition to remain constant while the others change, there are two different levels of setting the randomization; typically used to use the same evaluation partition with different training/validation partitions. One batch can have multiple RANDs.
- **attribute**: A piece of available information about a sample/patient; attribute refers to the type of information, not the value itself (ex. sex is an attribute, female is not)

### "FAQ"
 - **How can I add a new distribution option for partitions/steps?**
  Add the new distribution to the [subgroup distribution options file](subgroup_distribution_options.py). The distribution name cannot contain any spaces. Note that the distributions make use of the attribute_abbreviations dictionary at the bottom of the file. If you would like to add distributions that utilize attribute values not currently included in attribute_abbreviations, make sure to add them.
  _Do not make it so that one abbreviation is used for multiple attribute values_ (currently, the attribute abbreviations are only used to determine subgroup distributions, so while logical abbreviations are preferred, so long as each abbreviation is unique, it should not have any impact on output).
  Abbreviations of more than one letter are currently untested.
  
 - **How do I add a new repository or update and existing one?**
  The file path to the new directory summary file should be added in the [constants file](constants.py). For CXR datasets, a conversion file is expected as well. Certain formatting is expected, to see how the summary files are read, see the [summary_conversions file](utils/summary_conversion.py). Note that the dictionary keys are what must be passed to the `--repo` argument to use that repository, and thus must not contain any spaces.

  - **I encountered an error trying to use multiple steps and batches.** Multiple steps and batching should work together, however as the naming convention for the output partition files is slightly different with more than one step, loading batched partitions with more than one step may run into problems if any of the partition names contain a double underscore "__". 





