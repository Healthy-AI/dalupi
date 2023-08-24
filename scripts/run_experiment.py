import argparse
from dalupi.utils import load_config
from dalupi.utils import create_results_dir
from amhelpers.experiment import Experiment
from dalupi import ALL_SETTINGS

DEFAULT_TRAIN_SCRIPT_PATH = 'scripts/slurm_templates/training'
DEFAULT_PRE_SCRIPT_PATH = 'scripts/slurm_templates/preprocessing'
DEFAULT_POST_SCRIPT_PATH = 'scripts/slurm_templates/postprocessing'

DEFAULT_ACCOUNT = 'NAISS2023-5-242'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment.')
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--search_config_path', type=str)
    parser.add_argument('--settings', nargs='+', type=str, choices=ALL_SETTINGS, default=ALL_SETTINGS)
    parser.add_argument('--train_script', type=str, default=DEFAULT_TRAIN_SCRIPT_PATH)
    parser.add_argument('--pre_script', type=str, default=DEFAULT_PRE_SCRIPT_PATH)
    parser.add_argument('--post_script', type=str, default=DEFAULT_POST_SCRIPT_PATH)
    parser.add_argument('--account', type=str, default=DEFAULT_ACCOUNT)
    parser.add_argument('--gpu', type=str, choices=['V100', 'A40', 'A100'], default='A40')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    
    default_config = load_config(args.config_path)
    search_config = load_config(args.search_config_path) if args.search_config_path else None

    suffix = 'sweep' if args.sweep else '_'.join(args.settings)
    _, config = create_results_dir(default_config, suffix, update_config=True)

    experiment = Experiment(
        config,
        search_config,
        args.settings,
        args.train_script,
        args.pre_script,
        args.post_script,
        sweep=args.sweep,
        options={
            'account': args.account,
            'gpu': args.gpu,
        }
    )
    experiment.prepare()
    if not args.dry_run:
        experiment.run()
