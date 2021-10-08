


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

    logdir=os.path.join(root_logdir, *(run_id,wand_run_name))
    backupdir=os.path.join(root_backupdir, *(run_id,wand_run_name))
    
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
                queue='q64',
                walltime=time,
                local_directory='/scratch/$SLURM_JOB_ID',
                interface='ib0',
                scheduler_options={'interface':'ib0'},
                extra=["--lifetime", "50m"]
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


