





#%%
normalized=False
y_plus=15
target=['pr0.71_flux']
var=['u_vel',"pr0.71"]


def slice_loc(y_plus,var,target,normalized):
    """where to save the slices

    Args:
        y_plus (int): y_plus value of slice
        var (list): list of variables
        target (list): list of targets
        normalized (bool): if the data is normalized or not

    Returns:
        str: string of file save location
    """
    import os

    var_sort=sorted(var)
    var_string="_".join(var_sort)
    target_sort=sorted(target)
    target_string="_".join(target_sort)

    if normalized==True:
        slice_loc=os.path.join("/home/au643300/DataHandling/data/processed",'y_plus_'+str(y_plus)+"_VARS-"+var_string+"_TARGETS-"+target_string+"_normalized")
    else:
        slice_loc=os.path.join("/home/au643300/DataHandling/data/processed",'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string)

    return slice_loc

slice_loc(y_plus,var,target,normalized)



# %%
