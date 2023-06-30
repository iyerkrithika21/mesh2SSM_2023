# import shapeworks as sw
import glob 
import os
import trimesh
import time
import numpy as np
import multiprocessing as mp

class SurfaceDistance():
	"""This class calculates the symmetric vertex to surface distance of two
	trimesh meshes.
	"""

	def __init__(self):
		pass

	def __call__(self, A, B):
		"""
		Args:
		  A: trimesh mesh
		  B: trimesh mesh
		"""
		_, A_B_dist, _ = trimesh.proximity.closest_point(A, B.vertices)
		_, B_A_dist, _ = trimesh.proximity.closest_point(B, A.vertices)
		distance = .5 * np.array(A_B_dist).mean() + .5 * \
			np.array(B_A_dist).mean()

		return np.array([distance])

def calculate_surface_to_surface_distance(m,r):
	mesh = trimesh.load(m)
	

	

	s2sDist = SurfaceDistance()(mesh,trimesh.load(r))
	

	return np.mean(s2sDist)


def calculate_point_to_mesh_distance(m,p):
	mesh = trimesh.load(m)
	points = np.loadtxt(p)

	c = trimesh.proximity.ProximityQuery(mesh)
	p2mDist = c.signed_distance(points)

	
	

	return np.mean(p2mDist)