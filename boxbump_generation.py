import os
import vtk 
import numpy as np
import pyvista as pv
import shapeworks as sw
import glob



save_dir = "box_bump_100_test/"
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
mesh = pv.Box(level=6)
scale_x = 6
scale_y = 12
scale_z = 6
radius = 4.5
box = mesh.scale([scale_x, scale_y, scale_z], inplace=False).triangulate()
box.save(save_dir + "box.ply")

sphere = pv.Sphere(radius=radius,center=[0,0,0]).triangulate()
sphere.save(save_dir + "sphere.ply")

#---------------------------------------------------------------------------
# Read in shapeworks and remesh and smooth
#---------------------------------------------------------------------------
sw_box = sw.Mesh(save_dir + "box.ply")
sw_box.remesh(numVertices=2000, adaptivity=0.0).smooth(100,0.19).remesh(numVertices=2000, adaptivity=0.0)
sw_box.write(save_dir + "box_smoothed.ply")

sw_sphere = sw.Mesh(save_dir + "sphere.ply")
sw_sphere.remesh(numVertices=1000, adaptivity=0.0).smooth(50,0.01).remesh(numVertices=1000, adaptivity=0.0)
sw_sphere.write(save_dir + "sphere_smoothed.ply")



pv_box = pv.read(save_dir + "box_smoothed.ply")
pv_sphere = pv.read(save_dir + "sphere_smoothed.ply")
margin = 1
num_samples = 100
step = 2*(scale_y-radius-margin)/num_samples
starting_point = -1*(scale_y-radius-margin)
for i in range(num_samples):

	sphere = pv_sphere.translate((0,-1*(scale_y-radius-margin)+(i*step),scale_z), inplace=False)
	boxbump = sphere.boolean_union(pv_box)
	
	filename = "sample_" +str(i)+".ply"
	boxbump.save(save_dir + filename)

	# pl = pv.Plotter()
	# # _ = pl.add_mesh(box, color='tan', style='wireframe', line_width=3)
	# _ = pl.add_mesh(boxbump, color='b', style='wireframe', line_width=1)
	# pl.show_axes()
	# _ = pl.show_grid()
	# pl.show()
	del boxbump
	del sphere



