import os
import datetime
import re
import subprocess

BASE_SCRIPT_FILE = '/home/jingyihe/scratch/experiment_scripts_logging'
LOGGING_FILE = os.path.join(BASE_SCRIPT_FILE, '2019-06-14')


# class Logger:
#     def __init__(self, arg_lst):
#         self.log = self.declare(arg_lst)
#
#     def declare(self, args):
#         pattern = re.compile(r'[=]\d+')
#
#         search_fn = lambda x: pattern.findall(x)[0]
#         value_lst = map(search_fn, args)



if __name__ == '__main__':
    outfiles = os.listdir(os.path.join(LOGGING_FILE, '.out'))
    errfiles = os.listdir(os.path.join(LOGGING_FILE, '.err'))
    log_file = os.path.join(LOGGING_FILE, 'experiment_log.txt')
    argument_matcher = re.compile(r'[^_]*[^_]')

    # value_matcher = re.compile(r'[^=]\d+')
    # name_matcher = re.compile(r'\s+[^=]')

    with open(log_file, 'w') as wf:

        for f in outfiles:
            full_file_path = os.path.join(LOGGING_FILE, '.out', f)
            with open(full_file_path, 'r') as rf:
                last_line = rf.readlines()[-1]
            f = re.sub(r'-\d+\.out', '', f)
            out_line = ", ".join(f.split('_'))
            out_line += ", "
            out_line += last_line
            wf.writelines(out_line)






