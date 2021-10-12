

#%%

Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu


#Sigter efter en delta tau^+ på ca 30

delta_t_plus=30

delta_t=delta_t_plus*nu/(u_tau**(2))


#Så sætter delta_t til at være 2

