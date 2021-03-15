#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:55:35 2020

@author: au504946
"""

#%%




def get_valdata(quantity):
    import pandas as pd
    import numpy as np
    
    if quantity=='uTexa':
        data = pd.read_csv('data/external/validation/'+quantity+'_valData.csv', sep=' ', skiprows=1, header=None)
        data = data.dropna(axis=1)
    else:
        data = pd.read_csv('data/external/validation/'+quantity+'_valData.csv', sep=' ', skiprows=1, header=None)
        data = data.dropna(axis=1)
        data = data.drop(columns=[0])
        
    if quantity=='u':
        data.rename(columns={3: 'y+', 6:'u_mean', 9: 'uu+', 12:'ww+' }, inplace=True)
        Re = 14124
        RMS_adi =  np.sqrt(data['uu+'].to_numpy())
        mean_adi = data['u_mean'].to_numpy()
        
        data[quantity+'_plusmean']=mean_adi
        data[quantity+'_plusRMS']=RMS_adi
        data.drop(columns=['u_mean','uu+','ww+'],inplace=True)
        
     
    elif quantity=='pr1':
        data.rename(columns={3: 'y+', 6:'t_mean', 9: 't_rms+', 12:'ut+' }, inplace=True)
        Re = 14147
        RMS_adi =  data['t_rms+'].to_numpy()
        mean_adi = data['t_mean'].to_numpy()
        
        data[quantity+'_plusmean']=mean_adi
        data[quantity+'_plusRMS']=RMS_adi
        data.drop(columns=['u_mean','uu+','ww+'],inplace=True)
        
    elif quantity=='pr0.71':
        data.rename(columns={3: 'y+', 6:'t_mean', 9: 't_rms+', 12:'ut+' }, inplace=True)
        Re = 14062
        RMS_adi =  data['t_rms+'].to_numpy()
        mean_adi = data['t_mean'].to_numpy()
        
        data[quantity+'_plusmean']=mean_adi
        data[quantity+'_plusRMS']=RMS_adi
        data.drop(columns=['u_mean','uu+','ww+'],inplace=True)
        
    elif quantity=='pr0.025':
        data.rename(columns={3: 'y+', 6:'t_mean', 9: 't_rms+', 12:'ut+' }, inplace=True)
        Re = 14147
        RMS_adi =  data['t_rms+'].to_numpy()
        mean_adi = data['t_mean'].to_numpy()
        
        data[quantity+'_plusmean']=mean_adi
        data[quantity+'_plusRMS']=RMS_adi
        data.drop(columns=['u_mean','uu+','ww+'],inplace=True)
    
    if quantity=='uTexa':
        data.rename(columns={3: 'y', 6:'y+', 9: 'u_mean', 12:'ut+' }, inplace=True)
        RMS_adi = np.zeros(len(data['y']))
        #RMS_adi =  np.sqrt(data['ut+'].to_numpy())
        mean_adi = data['u_mean'].to_numpy() 
        
        data[quantity+'_plusmean']=mean_adi
        data[quantity+'_plusRMS']=RMS_adi
        data.drop(columns=['u_mean','uu+','ww+'],inplace=True)
    
    
    
    
    #ReTau = 395
    #utau=ReTau/Re
 
    
    return data







