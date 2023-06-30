
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau,StepLR
from data import Meshes,MeshesWithFaces

from mesh_vae import VAE
from model_autoencoder import Mesh2SSM_AE
import numpy as np
from torch.utils.data import DataLoader
# from util import cal_loss, IOStream, prepare_logger, cd_loss_L1
import sklearn.metrics as metrics
from chamfer_distance import ChamferDistance
from metrics import *
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
criterion = ChamferDistance()

def rank_zdims(args):
	training_data = MeshesWithFaces(directory = args.data_directory, extention=args.extention)
	args.scale = training_data.scale
	
	train_loader = DataLoader(training_data, num_workers=8,
							  batch_size=args.batch_size, shuffle=True, drop_last=True)
	test_data = MeshesWithFaces(directory = args.data_directory, extention=args.extention, partition ='test',k=args.k)
	args.test_scale = test_data.scale
	test_loader = DataLoader(test_data, num_workers=8,
							 batch_size=9, shuffle=False, drop_last=True)

	device = torch.device("cuda" if args.cuda else "cpu")
	args.device = device
	
	model = Mesh2SSM_AE(args)

	
	template = np.loadtxt(args.model_path + "best_template.particles")
	args.num_points = model.set_template(args,template)
	model.load_state_dict(torch.load(args.model_path + '/model.t7'))
	model = model.eval()
	model_vae = VAE(args).double().to(device)
	model_vae.load_state_dict(torch.load(args.model_path + '/model_vae.t7'))

	args.pred_dir = 'checkpoints/' + args.exp_name + "/output/"
	model.eval()
	model_vae.eval()
	
	print("Loaded models")
	
	
	args.num_points = model.set_template(args,template/args.scale)
	print("test_template")
	amu = torch.zeros(args.latent_dim)
	for data, idx, _, _ in train_loader:
		data =data.permute(0,2,1)
		particles, _ = model(data.to(args.device),idx.to(args.device))
		mu = model_vae.encoder(particles)[0]
		amu += mu.sum(0).detach().cpu()
	stdv = torch.zeros(args.latent_dim)
	print("amu calculation done")
	for data, idx, _, _ in train_loader:
		data =data.permute(0,2,1)
		particles, _ = model(data.to(args.device),idx.to(args.device))
		mu = model_vae.encoder(particles)[0].detach().cpu()
		stdv += (mu-amu).pow(2).sum(0).detach().cpu() / args.batch_size
	stdv = (stdv).sqrt().detach().cpu()
	std_idx = torch.argsort(stdv, descending=True)
	print("std calculation done")
	for data, idx, _, _ in train_loader:
		data =data.permute(0,2,1)
		particles, _ = model(data.to(args.device))#,idx.to(args.device))
		_,_,z_batch,x_recon = model_vae(particles)
		break
	
	print('most influential dimension:', std_idx)
	limit = 5
	inter = 5
	interpolation = torch.arange(-limit,limit+0.1,inter)
	row = std_idx[0]
	names = []
	for ids in range(3):
		row = std_idx[ids]
		z = z_batch[1,:]
		
		i=0
		for value in interpolation:
			z[row] = value			
			z_tensor = z
			sample = model_vae.decoder(z_tensor.double()).view(-1,3,args.num_points).permute(0, 2, 1)
			vae_r = sample.detach().cpu().numpy()*args.test_scale
			n = 'idx_'+str(row)+'_'+str(ids)+'_'+str(i)+".particles"
			names.append(n)
			np.savetxt(args.pred_dir + n, np.reshape(vae_r,(-1,3)))
			i=i+1
	print(names)
	return std_idx, stdv



def test(args):
	training_data = Meshes(directory = args.data_directory, extention=args.extention)
	args.scale = training_data.scale
	del training_data
	test_data = MeshesWithFaces(directory = args.data_directory, extention=args.extention, partition ='test',k=args.k)
	args.test_scale = test_data.scale
	test_loader = DataLoader(test_data, num_workers=8,
							 batch_size=9, shuffle=False, drop_last=True)

	device = torch.device("cuda" if args.cuda else "cpu")
	args.device = device
	
	model = Mesh2SSM_AE(args)
	
	try:
		template = np.loadtxt(args.model_path + "/best_template.txt")
	except:
		template = np.loadtxt(args.model_path + "/best_template.particles")
	args.num_points = model.set_template(args,template)#/args.scale)
	model.load_state_dict(torch.load(args.model_path + '/model.t7'))
	model = model.eval()
	model_vae = VAE(args).double().to(device)
	model_vae.load_state_dict(torch.load(args.model_path + '/model_vae.t7'))

	chamfer_dist = []
	for data, idx, label, names in test_loader:
		start = time.time()
		data, idx, label= data.to(device),idx.to(device),label.to(device).squeeze()
		
		data = data.permute(0, 2, 1)
		batch_size = data.size()[0]
		particles,reconstruction = model(data)
		end = time.time()
		print(end-start)
		dist1, dist2, idx1, idx2= criterion(label, particles)
		loss = 0.5 * (dist1.sqrt().mean() + dist2.sqrt().mean())
		chamfer_dist.append(loss.detach().item())
		for i in range(len(particles)):
			r =  particles[i].detach().cpu().numpy()*args.test_scale
			n = names[i].split(args.extention)[0] + ".particles"
			np.savetxt(args.recon_dir + n, r)

	print(f'Testing Chamfer Dist: {np.mean(chamfer_dist)} +/- {np.std(chamfer_dist)}')

	args.test_meshes = sorted(glob.glob(args.data_directory+"/test/*.ply"))
	args.test_particles  = sorted(glob.glob(args.recon_dir + "*.particles"))
	p2mDist = []
	
	for m,p in zip(args.test_meshes, args.test_particles):
		p2m = calculate_point_to_mesh_distance(m,p)
		p2mDist.append(p2m)
		

	print(f'Testing Point to Mesh Dist: {np.mean(p2mDist)} +/- {np.std(p2mDist)}')
	




if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Mesh2SSM: From surface meshes to statistical shape models of anatomy')
	parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N')
	parser.add_argument('--batch_size', type=int, default=10, metavar='batch_size',
						help='Size of batch)')
	parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
						help='Size of batch)')
	parser.add_argument('--epochs', type=int, default=100, metavar='N',
						help='number of epochs to train ')
	parser.add_argument('--use_sgd', type=bool, default=False,
						help='Use SGD')
	parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
						help='learning rate (default: 0.001, 0.1 if using sgd)')
	parser.add_argument('--vae_lr', type=float, default=0.001, metavar='LR',
						help='learning rate (default: 0.001, 0.1 if using sgd)')
	parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
						help='SGD momentum (default: 0.9)')
	parser.add_argument('--no_cuda', type=bool, default=False,
						help='enables CUDA training')
	parser.add_argument('--seed', type=int, default=42, metavar='S',
						help='random seed (default: 42)')
	parser.add_argument('--eval', type=bool,  default=False,
						help='evaluate the model')
	parser.add_argument('--dropout', type=float, default=0.5,
						help='dropout rate')
	parser.add_argument('--emb_dims', type=int, default=128, metavar='N',
						help='Dimension of embeddings of the mesh autoencoder for correspondence generation')
	parser.add_argument('--nf', type=int, default=8, metavar='N',
						help='Dimension of IMnet nf')
	parser.add_argument('--k', type=int, default=10, metavar='N',
						help='Num of nearest neighbors to use')
	parser.add_argument('--model_path', type=str, default='', metavar='N',
						help='Pretrained model path')
	parser.add_argument('--data_directory', type=str,
						help="data directory")
	parser.add_argument('--model_type', type=str, default = 'autoencoder',
						help="model type autoencoder or only encoder")
	parser.add_argument('--mse_weight', type=float, default=0.01, 
						help="weight for the mesh autoencoder(correspondence generation) mse reconstruction term in the loss")
	parser.add_argument('--template', type=str, default = "template",
						help="name of the template file")
	parser.add_argument('--extention', type=str, default=".ply",
						help="extention of the mesh files in the data directory")
	parser.add_argument('--gpuid', type=int, default=0,
						help="gpuid on which the code should be run")
	parser.add_argument('--vae_mse_weight', type=float, default=10,
						help="weight for the shape variational autoencoder(analysis) mse reconstruction term in the loss")
	parser.add_argument('--latent_dim', type = int, default = 64,
						help="latent dimensions of the shape variational autoencoder")
	args = parser.parse_args()
	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuid)
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	torch.manual_seed(args.seed)
	if args.cuda:
		print(
			'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
		torch.cuda.manual_seed(args.seed)
	else:
		print('Using CPU')

	

	args.checkpoint_dir = "checkpoints/" + args.exp_name
	args.model_path = args.checkpoint_dir + "/models/"
	args.recon_dir = args.checkpoint_dir+"/test_best/"
	if not os.path.exists(args.recon_dir):
		os.makedirs(args.recon_dir)
	test(args)
	rank_zdims(args)

