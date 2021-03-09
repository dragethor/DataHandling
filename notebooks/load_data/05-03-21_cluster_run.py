

from dask_jobqueue import SLURMCluster
from dask.distributed import Client

cluster=SLURMCluster(cores=2,
                     processes=1,
                     memory="5GB",
                     queue='q64',
                     walltime='0-00:45:00',
                     local_directory='/scratch/$SLURM_JOB_ID'
                     )


