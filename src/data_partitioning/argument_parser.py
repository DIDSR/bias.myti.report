from argparse import ArgumentParser
import pandas as pd
import sys

class PartitionArgumentParser():
    def __init__(self):
        self.parser = ArgumentParser()
        # Basic Args ==============================================================================
        self.parser.add_argument("-r", "--random-seed", dest="RAND", required=True, type=int,
          help="Random state used to generate partitions")
        self.parser.add_argument("--repository", "--repo", dest='repos', nargs='+', required=True, # TODO: update choices
          choices=['open_A1','BraTS'], help="list of repositories to be used")
        self.parser.add_argument("--betsy", default=False, action='store_true',                    # TODO: test betsy
          help="Pass to use betsy filepaths for summary and conversion files.")
        self.parser.add_argument("--id-column", "--id-col", dest='id_col', default='patient_id')   # TODO: help
        self.parser.add_argument("--save-location", "--save-loc", dest='save_loc', required=True)  
        self.parser.add_argument("--experiment-name", dest='experiment_name')                      # TODO: help
        self.parser.add_argument("--overwrite",default=False, action='store_true')                 # TODO: help
        self.parser.add_argument("--tasks", nargs="+", default=[])                                 # TODO: help
        
        # Attribute + Subgroup Args ===============================================================
        self.parser.add_argument("--attributes", '--att', dest='attributes', nargs='+', default=[],
          help="The attributes used to determine patient subgroups")
        self.parser.add_argument("--summary-attributes", dest='summary_attributes', nargs='+',
          default=[], help="Additioanl attributes to be included in summary outputs.")
        self.parser.add_argument("--subtract-from-smallest-subgroup", dest='subtract_from_smallest',
          default=5, type=int)
        self.parser.add_argument("--exclude-attributes", dest='exclude_attributes', nargs='+', default=[])
        
        # Partition Args ==========================================================================# TODO: help
        self.parser.add_argument("--partitions", nargs='+', required=True)
        self.parser.add_argument("--partition-sizes", dest='partition_sizes', nargs='+', required=True)
        self.parser.add_argument("--partition-distributions", dest='partition_distributions',
          nargs='+', default=['random'])
        
        # Step Args ===============================================================================# TODO: help
        self.parser.add_argument("--steps", default=1, type=int,
            help="Number of modification steps")
        self.parser.add_argument("--step-sizes",dest='step_sizes', default=[1], nargs='+')
        self.parser.add_argument("--step-distributions", dest='step_distributions', nargs='+',
          default=['random'])
        self.parser.add_argument("--constant-partitions", dest="constant_partitions",nargs="+", default=[], 
          help="Partitions for which to only generate a single step/keep constant across all steps")
        
        # Batch Args ==============================================================================# TODO: help
        self.parser.add_argument("--batch", nargs='+', default=[])
        self.parser.add_argument("--batch-rand", dest='batch_RAND', default=1, type=int)
        self.parser.add_argument("--accumulate", nargs='+', default=[], choices=['True','False'])
        self.parser.add_argument("--replace", nargs='+', default=[], choices=['True','False'])
        self.parser.add_argument("--replace-by-subgroup", dest='replace_by_subgroup', default=False, action='store_true')
        # WIP =====================================================================================# TODO
        self.parser.add_argument("--allow-not-reported", dest="allow_not_reported", default=False, action='store_true')
        self.parser.add_argument("--debug", default=False, action='store_true',
            help="Prints certain messages to assist in debugging unexpected behavior")
        self.parser.add_argument("--images-per-patient", dest='img_per_patient', default=0, type=int) # NOTE: 0 -> no restriction
        self.parser.add_argument("--image-limit-type", dest='img_limit_type',default='exact', choices=['exact', 'min', 'max'])
        self.parser.add_argument("--image-limit-method", dest='img_limit_method', default='PCR', choices=['PCR', 'first', 'last', 'random'])
        #self.parser.add_argument("--image-limit-allow-missing", dest='img_limit_allow_missing', default=False, action='store_true')
        self.parser.add_argument("--remove-null-PCR", dest="allow_null_PCR", default=True, action='store_false')
        self.parser.add_argument("--partition-summary-fig-type", dest='partition_summary_type', default='bar')
  
    def parse_args(self):
        args = self.parser.parse_args()
        # File Management =================================================================================
        if args.experiment_name is not None:
          args.save_loc = f"{args.save_loc}/{args.experiment_name}"
        if len(args.batch) != 0:
          args.batch_dir = f"{args.save_loc}/batch_{args.batch_RAND}"
          args.RAND_dir = f"{args.batch_dir}/RAND_{args.RAND}"
        else:
          args.RAND_dir = f"{args.save_loc}/RAND_{args.RAND}"
        
        args.exclude_attributes = {e.split(':')[0]:e.split(":")[-1].replace("_"," ").split(",") for e in args.exclude_attributes}
        # process list-style arguments
        args.repos = process_argument_list(args.repos)
        args.partitions = process_argument_list(args.partitions)
        args.partition_sizes = process_argument_list(args.partition_sizes)
        args.step_sizes = process_argument_list(args.step_sizes)
        
        # Check number of partitions and steps
        if len(args.partitions) != len(args.partition_sizes):
            raise Exception(f"Number of partitions ({len(args.partitions)}:  {args.partitions}) does not match number of partition sizes ({len(args.partition_sizes)}:  {args.partition_sizes})")
        args.partition_sizes = [float(x) for x in args.partition_sizes]
        args.partition_sizes = [x/sum(args.partition_sizes) for x in args.partition_sizes]
        
        args.partition_distributions = process_argument_list(args.partition_distributions)
        if len(args.partition_distributions) == 1:
            args.partition_distributions = args.partition_distributions * len(args.partitions)
        if len(args.partitions) != len(args.partition_distributions):
            raise Exception(f"Number of partitions ({len(args.partitions)}:  {args.partitions}) does not match number of partition distributions ({len(args.partition_distributions)}:  {args.partition_distributions})")
        
        if len(args.step_sizes) != args.steps:
            raise Exception(f"{len(args.step_sizes)} step sizes provided for {args.steps} steps")
        args.step_sizes = [float(x) for x in args.step_sizes]
        args.step_sizes = [x/sum(args.step_sizes) for x in args.step_sizes]
        
        args.step_distributions = process_argument_list(args.step_distributions)
        if len(args.step_distributions) == 1:
            args.step_distributions = args.step_distributions * args.steps
        if len(args.step_distributions) != args.steps:
            raise Exception(f"{len(args.step_distributions)} step distributions provided for {args.steps} steps")

        # process partition/step size information
        partition_information = pd.DataFrame(columns=['step', 'partition','size', "step_distribution", 'partition_distribution'])
        for i in range(args.steps):
            for ii, p in enumerate(args.partitions):
                size = float(args.partition_sizes[ii])*float(args.step_sizes[i])
                partition_information.loc[len(partition_information)] = [i, p, size, args.step_distributions[i], args.partition_distributions[ii]]
        partition_information['size'] /= partition_information['size'].sum()
        if args.debug: print(f"\nInput Partition information:\n{partition_information}")
        args.partition_information = partition_information

        # process batch arguments
        args.partition_RANDs = {p:args.RAND for p in args.partitions}
        for p in args.partitions:
            if p in args.batch:
                args.partition_RANDs[p] = args.batch_RAND
        
        # accumulate/replace
        if len(args.accumulate) == 1:
            args.accumulate = args.accumulate*len(args.partitions)

        if len(args.replace) == 1:
            args.replace = args.replace*len(args.partitions)

        args.attributes = list(set(args.attributes + args.tasks))

        return args

def process_argument_list(argument):
    """ Allows flexibility in how different list-style arguments are passed"""
    if len(argument) > 1:
        argument = ','.join(argument).replace(",,",",").replace("[","").replace("]","")
        argument = argument.split(",")
    return argument