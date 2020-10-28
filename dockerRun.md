### Pull docker image for mmdetection

#### Pull the base Image 
```sh
docker pull nasir6/mmdetection:base
```

#### To run 

```sh

# replace /home/ubuntu/code/ with path to the code directory

docker run -p 3000:3000 -v /home/ubuntu/code/:/home -it --runtime=nvidia --rm nasir6/mmdetection:base

cd mmdetection
conda env create -f environment.yml
python setup.py develop
# to test the synthesized classifier on MSCOCO. 
./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth 8 --dataset coco --out coco_results.pkl --zsd --syn_weights ../checkpoints/coco_65_15/classifier_best_137.pth

```

### Run Jupyter notebook

```ssh
ssh -L 3000:localhost:3000 ubuntu@[server-ip]
# run container bash and start jupyter notebook
nohup jupyter notebook --ip 0.0.0.0 --port 3000 --no-browser --allow-root &

```
