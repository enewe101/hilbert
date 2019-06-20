import os
import argparse
import re
import datetime

KYLIE_PROJECT_DIR = '/home/jingyihe/scratch/experiment_scripts_logging'
MULTISENSE_MLE_PY = '/home/jingyihe/Documents/word_sense/hilbert/runners/run_multisense.py'
EVALUATION_PY = '/home/jingyihe/Documents/hilbert_experiments/hilbert-experiments/evaluation/analysis/hilbert_eval.py'
DATA_PATH = '/home/jingyihe/Documents/evaluation_data/all_data.npz'


def add_arguments(parser):
    parser.add_argument('-ga', default=1, type=int, nargs='+', dest='gradient_accumulation',
                        help="provide a series of gradient accumulation steps.")
    parser.add_argument('-lr-scheduler', default=None, type=str, nargs='+', dest='lr_schedulers',
                        help="provide a series of scheduler")
    parser.add_argument('-slr', default=1e-4, type=float, nargs='+', dest='start_learning_rates',
                        help='give a series of start learning rates.')
    parser.add_argument('-u', default='10k', type=str, dest='num_updates',
                        help="give a number of updates, you can type (?)k or just a number")
    parser.add_argument('-p', default='350k', type=str, dest='batches',
                        help='number of batches, you can type (?)k or just a number')
    parser.add_argument('-o', default='/home/jingyihe/scratch/', type=str,
                        dest='output_dir', help='output base directory for storing the embeddings.')
    parser.add_argument('-b', default='/home/jingyihe/Documents/cooc/5w-dynamic-50k/', type=str, dest='cooc_dir',
                        help='cooccurrence directory')
    parser.add_argument('-temp', default=2, type=int, dest='temp',
                        help="temperature")
    parser.add_argument('-clip', default=None, type=float, nargs='+', dest='grad_clip',
                        help="gradient clipping norm")
    parser.add_argument('-K', default=5, type=int, dest='num_senses',
                        help="number of senses of a word")

def strNum_to_int(strNum):
    strNum = re.sub(r'k', '000', strNum)
    return int(strNum)


def write_python_command(**args):
    # init_emb = args['output_dir']+"50000"
    run_cmd = "python {} -b {} -o {} -l {} --writes=20 --updates={} --batch-size={}" \
              " --temperature={} -d 300 -thres 10 -K {}" \
              .format(MULTISENSE_MLE_PY,
                                args['cooc_dir'],
                                args['output_dir'],
                                args['lr'],
                                args['num_updates'],
                                args['num_batches'],
                                args['temp'],
                                args['num_senses']
                                # init_emb
                                )
    # base = "/".join(args['output_dir'].split('/')[:-1])
    # filename = args['output_dir'].split('/')[-1]

    # run_cmd += "\n\npython {} {} -e -v -s --base={} --data_path={}".format(EVALUATION_PY,
    #                                                                     filename,
    #                                                                     base,
    #                                                                    DATA_PATH)

    return run_cmd



def make_submit_script(submit_dir, job_name, cmd):
    job_file = os.path.join(submit_dir, "{}.job".format(job_name))

    with open(job_file,'w') as f:
        # some sbatch environment arguments
        f.writelines("#!/bin/bash\n")
        f.writelines("#SBATCH --account=def-dprecup\n")
        f.writelines("#SBATCH --time=3:00:00\n")
        f.writelines("#SBATCH --cpus-per-task=2\n")
        f.writelines("#SBATCH --mem-per-cpu=16G\n")
        f.writelines("#SBATCH --job-name={}\n".format(job_name))
        f.writelines("#SBATCH --output=.out/%x-%j.out\n")
        f.writelines("#SBATCH --error=.err/%x-%j.err\n")
        f.writelines("#SBATCH --gres=gpu:1\n")

        # virtualenv
        f.writelines("source /home/jingyihe/comp/bin/activate\n")

        # actual python command for running sample mle
        f.writelines(cmd)
        f.writelines("\n")


    # os.system("sbatch {}".format(job_file))



def main(submit_dir, args):
    ga = args['gradient_accumulation']
    lr_schedulers = args['lr_schedulers']
    start_lrs = args['start_learning_rates']
    num_updates = args['num_updates']
    num_batches = args['batches']
    output_base_dir = args['output_dir']
    cooc_dir = args['cooc_dir']
    temp = args['temp']
    grad_clip = args['grad_clip']
    num_senses = args['num_senses']
    # frac = args['frac']


    temp_str = 'temp={}'.format(temp)
    update_str = 'u={}'.format(num_updates)
    batch_str = 'b={}'.format(num_batches)
    sense_str = 'sense={}'.format(num_senses)
    # frac_str = 'frac={}'.format(frac)

    num_updates = strNum_to_int(num_updates)
    num_batches = strNum_to_int(num_batches)

    write_args = dict()
    write_args['cooc_dir'] = cooc_dir
    write_args['num_updates'] = num_updates
    write_args['num_batches'] = num_batches
    write_args['temp'] = temp
    write_args['num_senses'] = num_senses

    if lr_schedulers is None:
        lr_schedulers = [None]
    elif type(ga)!=list:
        lr_schedulers = [lr_schedulers]

    if ga is None:
        ga = [None]
    elif type(ga)!=list:
        ga = [ga]

    if start_lrs is None:
        start_lrs = [None]
    elif type(start_lrs)!=list:
        start_lrs = [start_lrs]

    if grad_clip is None:
        grad_clip = [None]
    elif type(grad_clip)!=list:
        grad_clip = [grad_clip]

    for lr in start_lrs:
        lr_str = 'slr={}'.format(lr)
        write_args['lr'] = lr

        for sche in lr_schedulers:
            sche_str = 'scheduler={}'.format(sche)
            write_args['sche'] = sche

            for g in ga:
                ga_str = 'ga={}'.format(g)
                write_args['g'] = g

                for gc in grad_clip:
                    gc_str = 'gc={}'.format(gc)
                    write_args['gc'] = gc

                    outfile_name = "_".join([lr_str, batch_str, update_str, temp_str, sense_str])
                    # outfile_name += "_balanced"
                    output_dir = os.path.join(output_base_dir, 'multisense_mle', outfile_name)
                    write_args['output_dir'] = output_dir
                    python_cmd = write_python_command(**write_args)
                    job_name = "_".join([lr_str, temp_str, sense_str])
                    # job_name += "multisense"
                    print("This job is made: ", job_name)
                    make_submit_script(submit_dir, job_name, python_cmd)
                    print("wrote a script... ")


if __name__ == '__main__':

    slurm_output_dir = datetime.date.today().isoformat()
    slurm_full_dir = os.path.join(KYLIE_PROJECT_DIR, slurm_output_dir)
    if not os.path.exists(slurm_full_dir):

        os.mkdir(slurm_full_dir)
        os.mkdir(os.path.join(slurm_full_dir,'.out'))
        os.mkdir(os.path.join(slurm_full_dir, '.err'))
        print("The following directory is made: ", slurm_full_dir)
        print("Slurm output and error will be written in .out and .err folder under the {}".format(slurm_full_dir))

    parser = argparse.ArgumentParser()

    add_arguments(parser)
    args = vars(parser.parse_args())

    main(slurm_full_dir, args)