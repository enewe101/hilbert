import argparse
import os
import datetime

KYLIE_PROJECT_DIR = '/home/jingyihe/scratch/experiment_scripts_logging'

def arguments_parser(parser):
    parser.add_argument('-i', type=str, dest='emb_dir', required=True,
                        help="directory that contains the embedding files")
    parser.add_argument('-b', type=str, dest='base_dir', required=True,
                        help="directory that contains all epochs of embedding files")

def write_job_scripts(submit_dir, test_name, run_command):
    job_file = os.path.join(submit_dir, "{}.job".format(test_name))
    with open(job_file, 'w') as f:
        # some sbatch environment arguments
        f.writelines("#!/bin/bash\n")
        f.writelines("#SBATCH --account=def-dprecup\n")
        f.writelines("#SBATCH --time=0:15:00\n")
        f.writelines("#SBATCH --cpus-per-task=1\n")
        f.writelines("#SBATCH --mem-per-cpu=16G\n")
        f.writelines("#SBATCH --job-name={}\n".format(test_name))
        f.writelines("#SBATCH --output=.out/%x-%j.out\n")
        f.writelines("#SBATCH --error=.err/%x-%j.err\n")
        f.writelines("#SBATCH --gres=gpu:1\n")

        # virtualenv
        f.writelines("source /home/jingyihe/comp/bin/activate\n")

        # actual python command for running sample mle
        f.writelines(run_command)
        f.writelines("\n")


def write_run_command(submit_dir, **args):
    for test in ['similarity',
                 'analogy',
                 'brown-pos',
                 'wsj-pos',
                 'sentiment',
                 'news',
                 'semcor-sst']:
        test_name = 'run_{}'.format(test)
        run_command = 'python /home/jingyihe/Documents/hilbert_experiments/hilbert-experiments/evaluation/run_experiments.py' \
                      '{} --base {}'.format(args['emb_dir'], args['base_dir'])
        run_command += test
        run_command += '--data-path /home/jingyihe/Documents/evaluation_data/all_data.npz '
        if test in ['wsj-pos',
                    'semcor-sst',
                    'brown-pos']:
            run_command += '--mb_size 16 --dropout 0.5 --n_layers 2 --rnn_hdim 128'
        else:
            run_command += '--ffnn --act_str relu --mb_size 64 --dropout 0.5 '\
                            '--model_str bilstm-max --rnn_hdim 128 --normalize'




def main():
    parser = argparse.ArgumentParser()
    arguments_parser(parser)
    args = vars(parser.parse_args())
    write_run_command(**args)


if __name__ == '__main__':
    slurm_output_dir = datetime.date.today().isoformat()
    slurm_full_dir = os.path.join(KYLIE_PROJECT_DIR, slurm_output_dir)
    if not os.path.exists(slurm_full_dir):
        os.mkdir(slurm_full_dir)
        os.mkdir(os.path.join(slurm_full_dir, '.out'))
        os.mkdir(os.path.join(slurm_full_dir, '.err'))
        print("The following directory is made: ", slurm_full_dir)
        print(
            "Slurm output and error will be written in .out and .err folder under the {}".format(
                slurm_full_dir))

    main()