# from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from random import randint
import torch 
from util import *

colors = ['#8a2244', '#da8c22', '#c687d5', '#80d6f8', '#440f06', '#000075', '#000000', '#e6194B', '#f58231', '#ffe119', '#bfef45']
colors2 = ['#02a92c', '#3a3075', '#3dde43', '#baa980', '#170eb8', '#f032e6', '#a9a9a9', '#fabebe', '#ffd8b1', '#fffac8', '#aaffc3']

dataset = 'voc'

# labels_to_plot = np.arange(21)
labels_to_plot = np.array([17,  18,  19,  20, 0])
# labels_to_plot = np.array([1,  2,  3,  4,  6,  8,  9, 10, 11, 12])

CLASSES = np.concatenate((['background'], get_class_labels(dataset)))

id_to_class = {idx: class_label for idx, class_label in enumerate(CLASSES)}

def tsne_plot_feats(f_feat, f_labels, path_save):
    # import pdb; pdb.set_trace()
    tsne = TSNE(n_components=2, random_state=0, verbose=True)
    syn_feature = np.load(f_feat)
    syn_label = np.load(f_labels)
    idx = np.where(np.isin(syn_label, labels_to_plot))[0]
    idx = np.random.permutation(idx)[0:2000]
    X_sub = syn_feature[idx]
    y_sub = syn_label[idx]
    # targets = np.unique(y)

    # colors = []
    for i in range(len(labels_to_plot)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    print(X_sub.shape, y_sub.shape, labels_to_plot.shape)

    X_2d = tsne.fit_transform(X_sub)
    fig = plt.figure(figsize=(6, 5))
    for i, c in zip(labels_to_plot, colors):
        plt.scatter(X_2d[y_sub == i, 0], X_2d[y_sub == i, 1], c=c, label=id_to_class[i][:3])
    plt.legend()
    # plt.show()
    fig.savefig(path_save)
    print(f"saved {path_save}")
    return X_sub, y_sub

def plot_unseen(epochs=20):
    # real_f, real_l = tsne_plot_feats('../data/voc/all_unseen_feats.npy', '../data/voc/all_unseen_labels.npy', 'plots/unseen_real_tsne.png')
    real_f, real_l = tsne_plot_feats('../data/voc/all_val_plot_feats.npy', '../data/voc/all_val_plot_labels.npy', 'plots_0.1/unseen_real_tsne_bg_0.1.png')
    print(f"len of real feats: {len(real_f)}")
    for epoch in range(0, epochs):
        f_feat = f'results/{epoch}_syn_feature.npy'
        f_labels = f'results/{epoch}_syn_label.npy'
        path_save = f'plots/{epoch}_unseen.png'
        syn_f, syn_l = tsne_plot_feats(f_feat, f_labels, path_save)
        print(f"len of syn feats: {len(syn_f)}")
        
        # merge and plot
        feats_all = np.concatenate((syn_f, real_f))
        label_all = np.concatenate((syn_l, real_l))
        tsne = TSNE(n_components=2, random_state=0, verbose=True)
        print(f"len of all feats: {len(feats_all)}")

        X_2d = tsne.fit_transform(feats_all)

        fig = plt.figure(figsize=(6, 5))
        # for i, c1, c2 in zip(labels_to_plot, colors, colors2): indx = np.where(label_all == i)[0]; plt.scatter(X_2d[indx[indx<5000], 0], X_2d[indx[indx<5000], 1], c=c1, label=f"s_{id_to_class[i][:3]}");plt.scatter(X_2d[indx[indx>=5000], 0], X_2d[indx[indx>=5000], 1], c=c2, label=f"r_{id_to_class[i][:3]}")
        for i, c1, c2 in zip(labels_to_plot, colors, colors2):
            indx = np.where(label_all == i)[0]
            plt.scatter(X_2d[indx[indx<2000],   0], X_2d[indx[indx<2000], 1], c=c1, label=f"s_{id_to_class[i][:5]}")
            plt.scatter(X_2d[indx[indx>=2000], 0], X_2d[indx[indx>=2000], 1], c=c2, label=f"r_{id_to_class[i][:5]}")
        plt.legend()
        fig.savefig(f'plots_merged/{epoch}_both.png')
        
        print(f"{epoch:02}/{epochs} ")

        plt.close('all')
        # import pdb; pdb.set_trace()

# plot_unseen(30)
# tsne_plot_feats('../data/voc/all_seen_feats.npy', '../data/voc/all_seen_labels.npy', 'plots/seen_real_tsne.png')
# def plot_seen():
# root = '../data/voc/'
# features = np.load(f"{root}/all_seen_feats.npy")
# labels = np.load(f"{root}/all_seen_labels.npy")

# rsync -avzzp jr1@172.31.20.60:/raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/plots_0.1* /home/nasir/Downloads/dgx_plots
# rsync -avzzp jr1@172.31.20.60:/raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/plots_0.1_merged/ /home/nasir/Downloads/plots_0.1_merged

# rsync -avzzp jr1@172.31.20.60:/raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/plots_0.1* /home/nasir/Downloads/dgx_plots/
