import argparse
from functools import partial 
from amhelpers.experiment import Postprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--experiment_path', type=str, required=True)
    args = parser.parse_args()

    def score_sorter(metric, df):
        mask = (df.subset == 'valid') & (df.domain == 'source')
        if mask.sum() == 0:
            mask = (df.subset == 'valid') & (df.domain == 'target')
        return df[mask][metric].item()

    if args.experiment == 'chestxray' or args.experiment == 'coco':
        metric = 'auc_average'
    elif args.experiment == 'mnist' or args.experiment == 'celeb':
        metric = 'accuracy_average'
    
    sorter = partial(score_sorter, metric)
    
    post = Postprocessing(args.experiment_path)
    post.collect_results(sorter)
