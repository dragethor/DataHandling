



""" 
Samler alle pdf og png filer fra models/output i figures mappen for nemmere at opdatere latex dokument
"""

#%%

import glob
from DataHandling import utility
import os
import shutil

name_list, _ = utility.get_runs_wandb()

output_path="/home/au643300/DataHandling/models/output"



for model in name_list:
    fig_folder_path="/home/au643300/DataHandling/reports/figures"
    fig_path=os.path.join(fig_folder_path,model)

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)


    full_dir=os.path.join(output_path,model)
    subdirs=os.listdir(full_dir)
    for subdir in subdirs:


        full_subdir=os.path.join(full_dir,subdir+"/")

        #Find all the pdf files
        pdf_files=glob.glob(full_subdir +"*.pdf")

        #Find all png files
        png_files=glob.glob(full_subdir +"*.png")

        
        all_files= pdf_files + png_files


        for file in all_files:
            file_split=file.split('/')
            file_name=file_split[-1]
            target=file_split[-2].split('-')[-1]

            new_name=file_name[:-4]+"-"+target[:-5]+file_name[-4:]

            new_fig_path=os.path.join(fig_path,new_name)

            shutil.copyfile(file,new_fig_path)
