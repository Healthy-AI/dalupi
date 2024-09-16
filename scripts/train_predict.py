import argparse
from dalupi.models.train_predict import train_model, predict_model
from dalupi.utils import load_config
from dalupi.utils import create_results_dir
from dalupi.data.utils import get_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--setting', type=str, required=True)
    parser.add_argument('--new_out_dir', action='store_true')
    parser.add_argument('--predict_only', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config_path)
    setting = args.setting
    experiment = config['experiment']

    if args.new_out_dir:
        assert not args.predict_only
        _, config = create_results_dir(config, args.setting, update_config=True)

    if experiment == 'chestxray' or experiment == 'coco':
        pipeline = get_pipeline(config)
        pipeline.split_data()
        pipeline.describe_test_data(config['results']['path'])
        
        valid_domain = 'target' if setting == 'target' else 'source'
        train_kwargs = {'pipeline': pipeline, 'subset': 'train_valid', 'valid_domain': valid_domain}
        
        if args.predict_only:
            model = None
        else:
            model = train_model(config, setting, **train_kwargs)
            if config[setting]['finetune']:
                model = train_model(config, setting, finetune=True, **train_kwargs)
        
        metrics = ['auc']
        predict_kwargs = {'pipeline': pipeline, 'metrics': metrics}
        
        predict_model(config, setting, model, subset='valid', prediction_domain=valid_domain, **predict_kwargs)
        predict_model(config, setting, model, subset='test', prediction_domain='target', **predict_kwargs)
        predict_model(config, setting, model, subset='test', prediction_domain='source', **predict_kwargs)
        
        pipeline.cleanup()
    
    elif experiment == 'mnist' or experiment == 'celeb':
        if args.predict_only:
            model = None
        else:
            model = train_model(config, setting)
            #if config[setting]['finetune']:
            #    model = train_model(config, setting, finetune=True)
        
        valid_domain = 'target' if setting == 'target' else 'source'

        metrics = ['accuracy']
        kwargs = {'metrics': metrics, 'plot_roc_curves': False}
        
        predict_model(config, setting, model, subset='valid', prediction_domain=valid_domain, **kwargs)
        predict_model(config, setting, model, subset='test', prediction_domain='target', **kwargs)
        predict_model(config, setting, model, subset='test', prediction_domain='source', **kwargs)
