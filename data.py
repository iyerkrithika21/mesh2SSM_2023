
import os
import sys
import glob
import pickle
import numpy as np
from torch.utils.data import Dataset
import pyvista as pv
from torch_geometric.utils import geodesic_distance
import torch
def download():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, 'data')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
		www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
		zipfile = os.path.basename(www)
		os.system('wget %s; unzip %s' % (www, zipfile))
		os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
		os.system('rm %s' % (zipfile))


def load_data(partition):
	download()
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, 'data')
	all_data = []
	all_label = []
	for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
		f = h5py.File(h5_name)
		data = f['data'][:].astype('float32')
		label = f['label'][:].astype('int64')
		f.close()
		all_data.append(data)
		all_label.append(label)
	all_data = np.concatenate(all_data, axis=0)
	all_label = np.concatenate(all_label, axis=0)
	return all_data, all_label


def translate_pointcloud(pointcloud):
	xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
	xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
	   
	translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
	return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
	N, C = pointcloud.shape
	pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
	return pointcloud


class ModelNet40(Dataset):
	def __init__(self, num_points, partition='train'):
		self.data, self.label = load_data(partition)
		self.num_points = num_points
		self.partition = partition        

	def __getitem__(self, item):
		pointcloud = self.data[item][:self.num_points]
		label = self.label[item]
		if self.partition == 'train':
			pointcloud = translate_pointcloud(pointcloud)
			np.random.shuffle(pointcloud)
		return pointcloud, label

	def __len__(self):
		return self.data.shape[0]
def save_json_gz(obj, filepath):
	import gzip
	import json

	json_str = json.dumps(obj)
	json_bytes = json_str.encode()
	with gzip.GzipFile(filepath, mode="w") as f:
		f.write(json_bytes)

def geodescis(pos, face, k):
	pos= torch.Tensor(pos)
	face = torch.Tensor(face)
	dist = -1*geodesic_distance(pos,face.t(),norm=False,num_workers=8)
	idx = dist.topk(k=k,dim=-1)[1]
	return idx

def load_meshes_with_faces(directory, partition, extention,k):
	files = sorted(glob.glob(directory +  partition +"/*"+extention))
	max_size = 0
	vertices_all = []
	pk_filename = directory + 'idx_' + str(k) +'_'+ partition + '.pkl'
	try:
		save = False
		with open(pk_filename, 'rb') as f:
			idx_all = pickle.load(f)
	except:
		save = True
		idx_all = {}
	
	max_scale = 0
	filename = []
	for f in files:
		mesh = pv.read(f)
		name = f.split("/")[-1]
		filename.append(name)
		vertices = np.array(mesh.points).astype('double')
		faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:]
		if (save == True):
			idx = geodescis(vertices, faces,k)
			idx_all[name] = idx

		
		scale = np.max(np.abs(vertices))
		if(scale>max_scale):
			max_scale = scale
		vertices_all.append(vertices)
		
		if (len(vertices)>max_size):
			max_size = len(vertices)
	if(save ==True):
		with open(pk_filename, 'wb') as f:
			pickle.dump(idx_all,f)

	return vertices_all, idx_all, max_size, max_scale, filename



class MeshesWithFaces(Dataset):
	def __init__(self, directory, partition='train',extention=".ply",k=10):
		
		self.data, self.idx_all, self.max_size, self.scale, self.filename = load_meshes_with_faces(directory, partition, extention,k)
		self.partition = partition        

	def __getitem__(self, item):
		name = self.filename[item]
		pointcloud = self.data[item]

		excess = self.max_size - len(pointcloud)
		list_idx = list(range(len(pointcloud)))
		if(excess > 0):
			repeat_idx = np.random.randint(0,len(pointcloud),excess)
			list_idx = list_idx + list(repeat_idx)

		pointcloud = pointcloud[list_idx,:]/self.scale
		
		
		label = pointcloud.copy()
		idx = self.idx_all[name]
		idx_extended = idx[list_idx]
		return pointcloud, idx_extended, label, name
		
	def __len__(self):
		return len(self.data)




def load_meshes(directory, partition, extention):
	files = sorted(glob.glob(directory +  partition +"/*"+extention))
	max_size = 0
	vertices_all = []
	
	max_scale = 0
	filename = []
	for f in files:
		mesh = pv.read(f)
		name = f.split("/")[-1]
		filename.append(name)
		vertices = np.array(mesh.points).astype('double')
		scale = np.max(vertices)
		if(scale>max_scale):
			max_scale = scale
		vertices_all.append(vertices)
		if (len(vertices)>max_size):
			max_size = len(vertices)
	return vertices_all, max_size, max_scale, filename



class Meshes(Dataset):
	def __init__(self, directory, partition='train',extention=".ply"):
		
		self.data, self.max_size, self.scale, self.filename = load_meshes(directory, partition, extention)
		self.partition = partition        

	def __getitem__(self, item):
		name = self.filename[item]
		pointcloud = self.data[item]
		excess = self.max_size - len(pointcloud)
		list_idx = list(range(len(pointcloud)))
		if(excess > 0):
			repeat_idx = np.random.randint(0,len(pointcloud),excess)
			list_idx = list_idx + list(repeat_idx)
		pointcloud = pointcloud[list_idx,:]/self.scale
		label = pointcloud.copy()

		return pointcloud, label, name

	def __len__(self):
		return len(self.data)
