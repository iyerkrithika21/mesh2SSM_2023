import getpass
from pathlib import Path
from swcc.api import swcc_session
from swcc.models import Dataset, Project

'''
Med Decathalon Dataset: Pancreas 
From the website: http://medicaldecathlon.com/
All data will be made available online with a permissive copyright-license (CC-BY-SA 4.0), allowing for data to be shared, distributed and improved upon. All data has been labeled and verified by an expert human rater, and with the best effort to mimic the accuracy required for clinical use. To cite this data, please refer to https://arxiv.org/abs/1902.09063
This dataset was pre-processing using ShapeWorks mesh grooming tools. 

Acknowledgements
If you use this pre-processed dataset in work that leads to published research, we humbly ask that you to cite ShapeWorks, add the following to the 'Acknowledgments' section of your paper:
"The National Institutes of Health supported this work under grant numbers NIBIB-U24EB029011, NIAMS-R01AR076120, NHLBI-R01HL135568, NIBIB-R01EB016701, and NIGMS-P41GM103545."
and add the following 'disclaimer': "The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health."

Citation
When referencing this dataset groomed with ShapeWorks, please include a bibliographical reference to the paper below, and, if possible, include a link to shapeworks.sci.utah.edu.
Joshua Cates, Shireen Elhabian, Ross Whitaker. "Shapeworks: particle-based shape correspondence and visualization software." Statistical Shape and Deformation Analysis. Academic Press, 2017. 257-298.

    @incollection{cates2017shapeworks,
    title = {Shapeworks: particle-based shape correspondence and visualization software},
    author = {Cates, Joshua and Elhabian, Shireen and Whitaker, Ross},
    booktitle = {Statistical Shape and Deformation Analysis},
    pages = {257--298},
    year = {2017},
    publisher = {Elsevier}
    }
 
'''

def public_server_download(download_dir):
	username = input('Enter username: ')
	password = getpass.getpass('Enter password: ')
	print("---------------------------------------------------------------------------------------")
	print("Please read the comments in the code about the license information before processing")
	print("---------------------------------------------------------------------------------------")
	input("Enter to proceed")
	with swcc_session() as public_server_session:
		try:
			public_server_session.login(username, password)
			dataset_name = 'MedDecathalon_Pancreas'

			dataset = Dataset.from_name(dataset_name)
		
			download_path = Path(download_dir)
			for project in dataset.projects:
				project.download(Path(download_path))
				break
		except:
			print("---------------------------------------------------------------------------------------")
			print("Please create an account on https://www.shapeworks-cloud.org/#/ to download the dataset")
			print("---------------------------------------------------------------------------------------")
			input("Enter to proceed")
		print("\n")

		print("You can also visualize the samples on the ShapeWorks Cloud portal https://www.shapeworks-cloud.org/#/ once you login")
		print("---------------------------------------------------------------------------------------")
		print("Please clone the dataset to modify the dataset. Do not edit the existing dataset")
		



public_server_download("./pancreas/")