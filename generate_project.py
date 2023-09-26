import shapeworks as sw 
import glob
import os 



def make_project_files(meshes,particles,data_dir,proj_name):
    # Create project spreadsheet
    project_location =  data_dir + "shape_models/"
    if not os.path.exists(project_location):
        os.makedirs(project_location)
    # Set subjects
    subjects = []
    print(len(meshes),len(particles))
    number_domains = 1
    for i in range(len(particles)):
        subject = sw.Subject()
        subject.set_number_of_domains(number_domains)
        rel_seg_files = sw.utils.get_relative_paths([meshes[i]], project_location)
        subject.set_original_filenames(rel_seg_files)
        subject.set_groomed_filenames(rel_seg_files)
        f = sw.utils.get_relative_paths([particles[i]], project_location)
        subject.set_local_particle_filenames(f)
        subject.set_world_particle_filenames(f)
        subjects.append(subject)
    project = sw.Project()
    project.set_subjects(subjects)
    spreadsheet_file = project_location + proj_name +".xlsx"
    project.save(spreadsheet_file) 




def main():
	meshes = sorted(glob.glob("path_to_mesh_files"))
	data_dir = "original_dir_containing_outputs"
	particles = sorted(glob.glob("path_to_predicted_particles_from_Mesh2SSM"))
	project_name = "output_project_name"
	make_project_files(meshes, particles, data_dir, project_name)
 

if __name__ == '__main__':
	main()
