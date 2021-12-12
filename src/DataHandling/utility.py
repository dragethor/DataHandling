

def get_runs_wandb():
    import wandb
    api = wandb.Api()
    entity, project = "dragethor", "Thesis"  # set to your entity and project
    runs = api.runs(entity + "/" + project)
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()})
            #if not k.startswith('_')})
        # .name is the human-readable name of the run.
        name_list.append(run.name)
    return name_list, config_list


def model_output_paths(model_name,y_plus,var,target,normalized):
    import os
    from DataHandling.features import slices

    model_path=os.path.join("/home/au643300/DataHandling/models/trained/",model_name)
    data_path=slices.slice_loc(y_plus,var,target,normalized)+"/"
    data_folder=os.path.basename(os.path.dirname(data_path))
    output_path='/home/au643300/DataHandling/models/output'
    output_path=os.path.join(output_path,model_name)
    output_path=os.path.join(output_path,data_folder)
    
    return model_path,output_path


def get_data(model_name,y_plus,var,target,normalized):
    """takes a TFrecord and returns list of features and targets for train, validation and test

    Args:
        data (TFrecord): list of TFrecord dataset

    Returns:
        (list, list, list): list of features, list of targets, and list of names
    """
    from DataHandling.features import slices
    import numpy as np
    from tensorflow import keras

    model_path,_=model_output_paths(model_name,y_plus,var,target,normalized)
    data=slices.load_validation(y_plus,var,target,normalized)

    feature_list=[]
    target_list=[]
    
    for data_type in data:
        feature_list.append(data_type[0])
        target_list.append(data_type[1].numpy())

    names=['train','validation','test']
    
    model=keras.models.load_model(model_path)
    predctions=[]
    for features in feature_list:
        predctions.append(model.predict(features))

    predctions=[np.squeeze(x,axis=3) for x in predctions]


    return feature_list,target_list,predctions,names




def get_run_dir(wand_run_name):
    """makes new dir for the run based on time of start and wandb run name

    Args:
        wand_run_name (str): name of run from command wandb.run.name

    Returns:
        str: two strings of dirs for log and backup
    """
    import time
    import os


    root_backupdir= os.path.join('/home/au643300/DataHandling/models', "backup")
    root_logdir = os.path.join('/home/au643300/DataHandling/models', "logs")
    run_id = time.strftime("run_%Y_%m_%d-%H_%M-")

    logdir=os.path.join(root_logdir, run_id+wand_run_name)
    backupdir=os.path.join(root_backupdir ,run_id+ wand_run_name)
    
    return logdir, backupdir


def slurm_q64(maximum_jobs,time='0-01:00:00',ram='50GB',cores=8):
    """Initiate a slurm cluster on q64

    Args:
        maximum_jobs (int): maxmimum number of jobs

    Returns:
        function handle: client instance of slurm
    """
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    
    cluster=SLURMCluster(cores=cores,
                memory=ram,
                queue='q64,q36,q24',
                walltime=time,
                local_directory='/scratch/$SLURM_JOB_ID',
                interface='ib0',
                scheduler_options={'interface':'ib0'}
                #extra=["--lifetime", "50m"]
                )
    client=Client(cluster)

    cluster.adapt(minimum_jobs=0,maximum_jobs=maximum_jobs)

    return client, cluster


def y_plus_to_y(y_plus):
    """Goes from specifed y_plus value to a y value

    Args:
        y_plus (int): value of y_plus to find the corresponding y value from

    Returns:
        int: The y value of the y_plus location
    """

    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    y= y_plus*nu/u_tau
    return y


def y_to_y_plus(y):
    """Goes from specifed y value to a y_plus value

    Args:
        y (int): value of y_plus to find the corresponding y value from

    Returns:
        int: The y value of the y_plus location
    """

    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    y_plus=y*u_tau/nu
    return y_plus


