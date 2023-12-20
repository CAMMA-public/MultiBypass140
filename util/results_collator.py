import os
import argparse
import numpy as np
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support as score

results_root = 'results/'
filename = 'results.csv'
header = 'no.,mode,task,acc,pr,re,f1,support\n'
modes = ['train', 'valid', 'test']
tasks = ['phase', 'step']

def compute_metrics(labels, preds):
    pred_labels = np.argmax(preds, axis=1) if preds.ndim > 1 else preds
    
    acc = np.sum(labels == pred_labels) * 100 / len(labels)
    acc = np.around(acc, 2)

    scores = score(labels, pred_labels)
    mean = np.mean(np.vstack(scores).T, axis=0)
    mean[:-1] *= 100
    mean = [acc] + np.around(mean, 2).tolist()

    std = np.std(np.vstack(scores).T, axis=0)
    std[:-1] *= 100
    std = [0.0] + np.around(std, 2).tolist()

    return mean, std

def compute_video_metrics(inds, preds, targets, score_fn):
    if len(preds) == 0: return [-1] * 5, [-1] * 5
    v_ids = np.array(list(map(lambda x: x.split('/')[-2], inds)))
    videos = np.unique(v_ids)
    vscores = []
    for vid in videos:
        idxs = np.argwhere(v_ids == vid).flatten()
        mean, _ = score_fn(np.array(targets)[idxs], np.array(preds)[idxs])
        vscores.append(mean[:])
    vmean = np.around(np.mean(vscores, axis=0), 2).tolist()
    vstd = np.around(np.std(vscores, axis=0), 2).tolist()
    return vmean, vstd


def generate_results(root_path, metrics='video'):
    for i, path in enumerate(sorted(Path(root_path).rglob('*.yaml'), key=lambda p: str(p))):
        exp_dir = path.parent
        results_file = os.path.join(results_root, str(exp_dir).split('models/')[-1], filename)
        print('computing results from:', exp_dir)
        results = header
        for j, mode in enumerate(modes):
            for task in tasks:
                re = [mode]
                re.append(task)
                try:
                    inds = np.load(os.path.join(exp_dir, '_'.join([mode, 'imgs.npy'])))
                    labels = np.load(os.path.join(exp_dir, '_'.join([mode, task, 'labels.npy'])))
                    preds = np.load(os.path.join(exp_dir, '_'.join([mode, task, 'preds.npy'])))
                    if metrics == 'image': mean, std = compute_metrics(labels, preds)
                    if metrics == 'video': mean, std = compute_video_metrics(inds, labels, preds, compute_metrics)
                    re += [ '$ ' +' \pm '.join([str(m), str(std[k])]) + ' $' for k, m in enumerate(mean)]
                except Exception as e:
                    re += ['-1.00 \pm -1.00'] * 5
                results += ','.join(map(str, [j] + re)) + '\n'
        if not os.path.exists(os.path.dirname(results_file)):
            os.makedirs(os.path.dirname(results_file))
        with open(results_file, 'w') as fp:
            fp.write(results)
        print('computing results: done!!!')
    return


def Main(args):
    print('*'*50)
    print('search folder     :', args.experiment)
    print('collecting metrics:', args.metrics)
    print('*'*50)
    
    generate_results(args.experiment, metrics=args.metrics)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse model training options')
    parser.add_argument('-e', '--experiment', default='',
                    help='Root path to experiments folder')

    parser.add_argument('-m', '--metrics', default='video',
                    help='type of metrics to compute: image/video')

    args = parser.parse_args()
    Main(args)
