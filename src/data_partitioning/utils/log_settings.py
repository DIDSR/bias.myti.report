from datetime import datetime
import json
import sys
import os

def log_partition_settings(args):
    """ Logs the settings needed for recrations, as well as the current date and time. """
    info_dict = {}
    command = ['python'] + sys.argv
    info_dict['command'] = " ".join(command)
    info_dict['venv'] = sys.prefix
    info_dict['created on'] = datetime.today().strftime('%Y-%m-%d %H:%M')
    log_file = os.path.join(args.RAND_dir, 'partition_settings.log')
    with open(log_file, 'w') as fp:
        json.dump(info_dict, fp, indent=0, separators=["\n",":\n\t"])
    