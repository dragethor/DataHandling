
#%%


api = wandb.Api()
entity, project = "dragethor", "Thesis"  # set to your entity and project 
runs = api.runs(entity + "/" + project) 
summary_list, config_list, name_list = [], [], []


for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    if run.state == "finished":
        summary_list.append(run.summary._json_dict)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()})
            #if not k.startswith('_')})
        # .name is the human-readable name of the run.
        name_list.append(run.name)


slurm_arrary_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
