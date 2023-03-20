#[MATE: Masked Autoencoders are Online 3D Test-Time Learners](https://arxiv.org/abs/2211.11432#:~:text=We%20propose%20MATE%2C%20the%20first,not%20be%20anticipated%20during%20training.)
MATE is the first 3D Test-Time Training (TTT) method which makes 3D object recognition architectures robust to 
distribution shifts which can commonly occur in 3D point clouds. 
MATE follows the classical TTT paradigm of using an auxiliary objective to make the network robust to 
distribution shifts at test-time. 
To this end, MATE employs the self-supervised test-time objective of reconstructing aggressively masked 
input point cloud patches.

In this repository we provide our pre-trained models and codebase to reproduce the results reported in our 
paper. 
## Requirements
```
PyTorch >= 1.7.0 < 1.11.0  
python >= 3.7  
CUDA >= 9.0  
GCC >= 4.9  
```
To install all additional requirements (open command line and run):
```
pip install -r requirements.txt

cd ./extensions/chamfer_dist
python setup.py install --user

cd ..

cd ./extensions/emd
python setup.py install --user
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Data Preparation
Our code currently supports three different datasets: [ModelNet40](https://arxiv.org/abs/1406.5670), [ShapeNetCore](https://arxiv.org/abs/1512.03012) and [ScanObjectNN](https://arxiv.org/abs/1908.04616).
  
### Download
To use these datasets with our code, first download them from the following sources:  
- [ModelNet](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) 

- [ShapeNetCore](https://cloud.tsinghua.edu.cn/f/06a3c383dc474179b97d/)

- [ScanObjectNN](https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip) (It is necessary to first agree to the terms of use [here](https://forms.gle/g29a6qSgjatjb1vZ6))  

Then, extract all of these folders into the same directory for easier use.

### Adding corruptions to the data
To add distribution shifts to the data, corruptions from [ModelNet40-C](https://arxiv.org/abs/2201.12296) are used.  
For experiments on corrupted ModelNet data, the ModelNet40-C dataset can be downloaded [here](https://drive.google.com/drive/folders/10YeQRh92r_WdL-Dnog2zQfFr03UW4qXX).  
Compute the same corruptions for ShapeNetCore and ScanObjectNN, if needed.

```
python ./datasets/create_corrupted_dataset.py --main_path <path/to/dataset/parent/directory> --dataset <dataset_name>
```
Replace `<dataset_name>` with either `scanobjectnn` or `shapenet` as required. 

Note that for computation of the corruptions "occlusion" and "lidar", model 
meshes are needed. These are computed with 
the [open3d](http://www.open3d.org/docs/release/getting_started.html) library. 

## Obtaining Pre-Trained Models
All our pretrained models are available at 
this [Google-Drive](https://drive.google.com/drive/folders/1TR46XXp63rtKxH5ufdbfI-X0ZXx8MyKm?usp=share_link).

The `jt` models are jointly trained for reconstruction and classification, `src_only` 
models are trained for only the classification task.  

## Test-Time-Training (TTT)
### Setting data paths 
For TTT, go to `cfgs/tta/tta_<dataset_name>.yaml` and set the `tta_dataset_path` variable to the relative path of the dataset parent directory.  
E.g. if your data for ModelNet-C is in `./data/tta_datasets/modelnet-c`, set the variable to `./data/tta_datasets`.  

A jointly trained model can be used for test-time training by:  
```
CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name <dataset_name> --online --grad_steps 1 --config cfgs/tta/tta_<dataset_name>.yaml --ckpts <path/to/pretrained/model>
```
This will run the `TTT-Online (for one gradient step)`.

For running the `TTT-Standard`, following command can be used: 
```
CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name <dataset_name> --grad_steps 20 --config cfgs/tta/tta_<dataset_name>.yaml --ckpts <path/to/pretrained/model>
```

## Training Models
### Setting data paths
To train a new model on one of the three datasets, go to `cfgs/dataset_configs/<dataset_name>.yaml` and set the `DATA_PATH` 
variable in the file to the relative path of the dataset folder.  

### Running training scripts
After setting the paths, a model can be jointly trained by
```
CUDA_VISIBLE_DEVICES=0 python train.py --jt --config cfgs/pre_train/pretrain_<dataset_name>.yaml --dataset <dataset_name>
```  
A model for a supervised only baseline can be trained by
```
CUDA_VISIBLE_DEVICES=0 python train.py --only_cls --config cfgs/pre_train/pretrain_<dataset_name>.yaml --dataset <dataset_name>
```  
The trained models can then be found in the corresponding `experiments` subfolder.

## Inference

For a basic inference baseline without adaptation, use
```
CUDA_VISIBLE_DEVICES=0 python test.py --dataset_name <dataset_name> --config cfgs/pre_train/pretrain_<dataset_name>.yaml  --ckpts <path/to/pretrained/model> --test_source
```
Scripts for pretraining, testing and test-time training can also be found in `commands.sh`.

#### To cite us: 
```bibtex
@InProceedings{mirza2023mate,
    author    = {Mirza, M. Jehanzeb and Shin, Inkyu and Lin, Wei and Schriebl, Andreas and Sun, Kunyang and
                 Choe, Jaesung and Kozinski, Mateusz and Possegger, Horst and Kweon, In So and Yoon, Kun-Jin and Bischof, Horst},
    title     = {MATE: Masked Autoencoders are Online 3D Test-Time Learners},
    booktitle = {arXiv preprint arXiv:2211.11432},
    year      = {2023}
}
```
We acknowledge [PointMAE](https://github.com/Pang-Yatian/Point-MAE) for their open source implementation.
