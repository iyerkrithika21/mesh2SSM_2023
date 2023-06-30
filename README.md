# Mesh2SSM

Implementation of the paper ["Mesh2ssm: From surface meshes to statistical shape models of anatomy"](https://arxiv.org/abs/2305.07805). 
Please cite this paper if you use the code. 


## Running the code

### Dataset
Arrange the input meshes in three folders - `train, test, val`
These meshes need to be pre-processed, i.e., centered, aligned, smoothed, and should roughly contain the same number of vertices and edges. 
Each dataset should also contain the template point cloud in text format placed outside of the train, test, and val folders. Example template file format where we have the x,y,z position of the point, and each row is a correspondence point. 
```
14.2509    5.10197    17.3891
61.679    10.9403    11.4656
70.2814    15.9074    12.156
-54.3031    -23.9498    -32.8051
-16.3315    -13.9949    -1.55633
62.0011    29.5519    29.7298
```

### Training and Testing
```
python train_geodesic.py [--with appropriate tags]
python test.py [--with appropriate tags]
```

#### Usage
```
usage: train_geodesic.py [-h] [--exp_name N] [--dataset N] [--batch_size batch_size] [--test_batch_size batch_size] [--epochs N]
                         [--use_sgd USE_SGD] [--lr LR] [--vae_lr LR] [--momentum M] [--no_cuda NO_CUDA] [--seed S] 
                         [--dropout DROPOUT] [--emb_dims N] [--nf N] [--k N] [--model_path N] [--data_directory DATA_DIRECTORY]
                         [--model_type MODEL_TYPE] [--mse_weight MSE_WEIGHT] [--template TEMPLATE] [--extention EXTENTION]
                         [--gpuid GPUID] [--vae_mse_weight VAE_MSE_WEIGHT] [--latent_dim LATENT_DIM]

Mesh2SSM: From surface meshes to statistical shape models of anatomy

arguments:
  -h, --help            show this help message and exit
  --exp_name N          Name of the experiment
  --dataset N
  --batch_size batch_size
                        Size of batch)
  --test_batch_size batch_size
                        Size of batch)
  --epochs N            number of epochs to train
  --use_sgd USE_SGD     Use SGD
  --lr LR               learning rate (default: 0.001, 0.1 if using sgd)
  --vae_lr LR           learning rate (default: 0.001, 0.1 if using sgd)
  --momentum M          SGD momentum (default: 0.9)
  --no_cuda NO_CUDA     enables CUDA training
  --seed S              random seed (default: 42)
  --dropout DROPOUT     dropout rate
  --emb_dims N          Dimension of embeddings of the mesh autoencoder for correspondence generation
  --nf N                Dimension of IMnet nf
  --k N                 Num of nearest neighbors to use
  --model_path N        Pretrained model path
  --data_directory DATA_DIRECTORY
                        data directory
  --model_type MODEL_TYPE
                        model type autoencoder or only encoder
  --mse_weight MSE_WEIGHT
                        weight for the mesh autoencoder(correspondence generation) mse reconstruction term in the loss
  --template TEMPLATE   name of the template file
  --extention EXTENTION
                        extention of the mesh files in the data directory
  --gpuid GPUID         gpuid on which the code should be run
  --vae_mse_weight VAE_MSE_WEIGHT
                        weight for the shape variational autoencoder(analysis) mse reconstruction term in the loss
  --latent_dim LATENT_DIM
                        latent dimensions of the shape variational autoencoder
```
