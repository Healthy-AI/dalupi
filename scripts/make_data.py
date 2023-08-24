import argparse
from dalupi.utils import load_config
from os.path import join

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make dataset.')
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_path)
    experiment = config['experiment']

    if experiment == 'chestxray':
        from dalupi.data.chestxray import prepare_datasets
        source_dir = config['data']['source_path']
        target_dir = config['data']['target_path']
        prepare_datasets(source_dir, target_dir)
    
    elif experiment == 'coco':
        from dalupi.data.coco import prepare_datasets
        from dalupi.data.coco import remove_class_images, remove_grayscale_images

        super_cats_source = ['indoor', 'appliance']  # source = indoor
        cats_source = []

        super_cats_target = ['outdoor', 'vehicle']  # target = outdoor
        cats_target = []

        cats_pred = ['person', 'cat', 'dog', 'bird']

        n_background = 1000

        data_source, data_target, pi_source, pi_target = prepare_datasets(
            super_cats_source,
            cats_source,
            super_cats_target,
            cats_target,
            cats_pred,
            n_background,
            dataset_path=config['data']['image_path'],
            version=2017,
            seed=config['seed']
        )

        n_only_persons = 2500

        data_source, pi_source = remove_class_images(
            data_source,
            pi_source,
            cats_pred,
            'person',
            n_only_persons,
            seed=config['seed']
        ) 

        data_target, pi_target = remove_class_images(
            data_target,
            pi_target,
            cats_pred,
            'person',
            n_only_persons,
            seed=config['seed']
        )

        data_source, pi_source = remove_grayscale_images(data_source, pi_source)
        data_target, pi_target = remove_grayscale_images(data_target, pi_target)

        data_path = config['data']['csv_path']
        
        data_source.to_csv(join(data_path, 'data_source.csv'), index=False)
        pi_source.to_csv(join(data_path, 'pi_source.csv'), index=False)
        
        data_target.to_csv(join(data_path, 'data_target.csv'), index=False)
        pi_target.to_csv(join(data_path, 'pi_target.csv'), index=False)
    
    elif experiment == 'mnist':
        from dalupi.data.mnist.utils import load_or_make_dataset

        for skew in [0.2, 0.4, 0.6, 0.8, 1.0]:
            config['data']['skew'] = skew
            load_or_make_dataset(config)
