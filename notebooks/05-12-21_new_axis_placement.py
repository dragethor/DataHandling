

#%%

import numpy as np

Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu

x_range=12/2
z_range=6/2

gridpoints_x=int(255/2)+1
gridponts_z=int(255/2)+1


x_plus_max=x_range*u_tau/nu
z_plus_max=z_range*u_tau/nu


x_plus_max=np.round(x_plus_max).astype(int)
z_plus_max=np.round(z_plus_max).astype(int)

axis_range_x=np.array([0,470,950,1420,1900,2370])
axis_range_z=np.array([0,295,590,890,1185])


placement_x=axis_range_x*nu/u_tau
placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)


placement_z=axis_range_z*nu/u_tau
placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)


print(placement_z)