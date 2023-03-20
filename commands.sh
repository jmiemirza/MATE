#Pretraining
CUDA_VISIBLE_DEVICES=0 python train.py --jt --config cfgs/pre_train/pretrain_modelnet.yaml --dataset modelnet
CUDA_VISIBLE_DEVICES=0 python train.py --only_cls --config cfgs/pre_train/pretrain_modelnet.yaml --dataset modelnet

CUDA_VISIBLE_DEVICES=0 python train.py --jt --config cfgs/pre_train/pretrain_scanobject.yaml --dataset scanobject_nbg --ckpts models/pretrain.pth
CUDA_VISIBLE_DEVICES=0 python train.py --only_cls --config cfgs/pre_train/pretrain_scanobject.yaml --dataset scanobject_nbg --ckpts models/pretrain.pth

CUDA_VISIBLE_DEVICES=0 python train.py --jt --config cfgs/pre_train/pretrain_shapenetcore.yaml --dataset shapenetcore
CUDA_VISIBLE_DEVICES=0 python train.py --only_cls --config cfgs/pre_train/pretrain_shapenetcore.yaml --dataset shapenetcore

#TTT - Online
CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name modelnet --online --grad_steps 1 --config cfgs/tta/tta_modelnet.yaml --ckpts models/modelnet_jt.pth

CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name scanobject --online --grad_steps 1 --config cfgs/tta/tta_scanobj.yaml --ckpts models/scanobject_jt.pth

CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name shapenetcore --online --grad_steps 1 --config cfgs/tta/tta_shapenet.yaml --ckpts models/shapenet_jt.pth

#TTT - Standard
CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name modelnet --grad_steps 20 --config cfgs/tta/tta_modelnet.yaml --ckpts models/modelnet_jt.pth

CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name scanobject --grad_steps 20 --config cfgs/tta/tta_scanobj.yaml --ckpts models/scanobject_jt.pth

CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name shapenetcore --grad_steps 20 --config cfgs/tta/tta_shapenet.yaml --ckpts models/shapenet_jt.pth


#Inference only
CUDA_VISIBLE_DEVICES=0 python test.py --dataset_name modelnet --config cfgs/tta/tta_modelnet.yaml --ckpts models/modelnet_src_only.pth --test_source
CUDA_VISIBLE_DEVICES=0 python test.py --dataset_name modelnet --config cfgs/tta/tta_modelnet.yaml --ckpts models/modelnet_jt.pth --test_source

CUDA_VISIBLE_DEVICES=0 python test.py --dataset_name scanobject --config cfgs/tta/tta_scanobj.yaml --ckpts models/scanobject_src_only.pth --test_source
CUDA_VISIBLE_DEVICES=0 python test.py --dataset_name scanobject --config cfgs/tta/tta_scanobj.yaml --ckpts models/scanobject_jt.pth --test_source

CUDA_VISIBLE_DEVICES=0 python test.py --dataset_name shapenetcore --config cfgs/tta/tta_shapenet.yaml --ckpts models/shapenet_src_only.pth --test_source
CUDA_VISIBLE_DEVICES=0 python test.py --dataset_name shapenetcore --config cfgs/tta/tta_shapenet.yaml --ckpts models/shapenet_jt.pth --test_source
