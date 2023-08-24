import os
import argparse
import subprocess
from os.path import join
from dalupi import ALL_SETTINGS
from amhelpers.experiment import _create_jobscript_from_template

DEFAULT_TRAIN_SCRIPT_PATH = 'scripts/slurm_templates/training'

DEFAULT_ACCOUNT = 'NAISS2023-5-242'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--settings', nargs='+', type=str, choices=ALL_SETTINGS, default=ALL_SETTINGS)
    parser.add_argument('--train_script', type=str, default=DEFAULT_TRAIN_SCRIPT_PATH)
    parser.add_argument('--account', type=str, default=DEFAULT_ACCOUNT)
    parser.add_argument('--gpu', type=str, choices=['V100', 'A40', 'A100'], default='A40')
    args = parser.parse_args()

    jobdir = os.path.join(args.experiment_path, 'jobscripts')

    for setting in args.settings:
        command = ['sbatch']

        configs = os.listdir(join(args.experiment_path, '%s_configs' % setting))
        n_jobs = len(configs)
        if n_jobs > 1:
            command.append('--array=1-%d' % n_jobs)
        
        jobscript_path = _create_jobscript_from_template(
            template=args.train_script,
            experiment='',
            experiment_path=args.experiment_path,
            setting=setting,
            jobname='job_predict_%s' % setting,
            jobdir=jobdir,
            options={
                'account': args.account,
                'gpu': args.gpu,
            }
        )
        
        with open(jobscript_path, 'r') as f:
            jobscript = f.read()
        
        jobscript = jobscript.strip() + ' --predict_only\n'
        
        with open(jobscript_path, 'w') as f:
            f.write(jobscript)
        
        command.append(jobscript_path)
        subprocess.run(command)
