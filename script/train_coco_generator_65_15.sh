python trainer.py --manualSeed 806 \
--cls_weight 0.0 --cls_weight_unseen 1.00 --nclass_all 81 --syn_num 250 --val_every 1 \
--cuda --netG_name MLP_G --netD_name MLP_D \
--nepoch 30 --nepoch_cls 18 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--dataset coco --batch_size 32 --nz 300 --attSize 300 --resSize 1024 --gan_epoch_budget 38000 \
--lr 0.0001 --lr_step 30 --lr_cls 0.0001 \
--pretrain_classifier mmdetection/work_dirs/coco_con_with_noise_l2/epoch_12.pth \
--pretrain_classifier_unseen coco_65_15_fc_1024_300.pth \
--class_embedding MSCOCO/fasttext.npy \
--testsplit test_default_0.6_0.3 \
--trainsplit train_default_0.6_0.3 \
--classes_split 65_15 \
--lz_ratio 0.0 \
--outname checkpoints/coco_65_15_default_noise_0.05 \

# --netD checkpoints/coco_65_15_0.6_1_0_1/disc_latest.pth \
# --netG checkpoints/coco_65_15_0.6_1_0_1/gen_latest.pth
# split, th, ls, lu, lz test_default_0, test_con_0
