from plot import plot_acc, plot_gan_losses, plot_confusion_matrix
from arguments import parse_args
import random
import torch
import torch.backends.cudnn as cudnn
import os
import numpy as np
from dataset import FeaturesCls, FeaturesGAN
from train_cls import TrainCls
from train_gan import TrainGAN
from generate import load_unseen_att, load_all_att, load_seen_att, load_unseen_att_with_bg
from mmdetection.splits import get_class_labels

opt = parse_args()
for arg in vars(opt): print(f"######################  {arg}: {getattr(opt, arg)}")

"""
"""

try:
    os.makedirs(opt.outname)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
# init classifier
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

unseen_attributes, unseen_att_labels = load_unseen_att(opt)
attributes, _ =load_all_att(opt)
trainCls = TrainCls(opt, att=None)

print('initializing GAN Trainer')
trainFGGAN = TrainGAN(opt, attributes, unseen_attributes, unseen_att_labels, gen_type='FG')
start_epoch = 0

if opt.netD and opt.netG:
    # resume from checkpoint
    start_epoch = trainFGGAN.load_checkpoint()

# real features dataset
seenDataset = FeaturesGAN(opt)

for epoch in range(start_epoch, opt.nepoch):
    # randomly sample subset of real features 
    real_features, real_labels = seenDataset.epochData(include_bg=True)
    # skip GAN train 
    # trainFGGAN(epoch, features, labels)
    # synthesize features
    syn_feature, syn_label = trainFGGAN.generate_syn_feature(unseen_att_labels, unseen_attributes, num=opt.syn_num)
    # num_of_bg = opt.syn_num*4
    # real_feature_bg, real_label_bg = seenDataset.getBGfeats(num_of_bg)#all_features[bg_inds], all_labels[bg_inds]
    
    # replace some of the real bg features with random values 
    # num_random_bg = len(real_label_bg) // 2
    # real_feature_bg[:num_random_bg] = np.random.rand(num_random_bg, 1024)

    # concatenate synthesized + real bg features
    train_feature = np.concatenate((syn_feature.data.numpy(), real_features))
    train_label = np.concatenate((syn_label.data.numpy(), real_labels))
    
    trainCls(train_feature, train_label, gan_epoch=epoch)

    # -----------------------------------------------------------------------------------------------------------------------
    # plots
    classes = np.concatenate((['background'], get_class_labels(opt.dataset)))
    plot_confusion_matrix(np.load(f'{opt.outname}/confusion_matrix_Train.npy'), classes, classes, opt, dataset='Train', prefix=opt.class_embedding.split('/')[-1])
    plot_confusion_matrix(np.load(f'{opt.outname}/confusion_matrix_Test.npy'), classes, classes, opt, dataset='Test', prefix=opt.class_embedding.split('/')[-1])
    plot_acc(np.vstack(trainCls.val_accuracies), opt, prefix=opt.class_embedding.split('/')[-1])

    # save models
    if trainCls.isBestIter == True:
        trainFGGAN.save_checkpoint(state='best')

    trainFGGAN.save_checkpoint(state='latest')
