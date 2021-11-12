

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sigmoid = lambda x: 1/(1+np.exp(-x))

relu = lambda x: np.maximum(0,x)

def elu(x,alpha):
    return np.piecewise(x , [ x<= 0],[lambda x: alpha*(np.exp(x)-1),lambda x: x])


x_val=np.linspace(-3, 3)



plt.figure()
plt.plot(x_val,elu(x_val,1),label="ELU")
plt.plot(x_val,sigmoid(x_val),label="Sigmoid")
plt.plot(x_val,relu(x_val),label="ReLU")
plt.xlabel('x')
plt.ylabel('g(x)')
plt.legend()
plt.savefig('/home/au643300/DataHandling/reports/figures/activation.pdf',bbox_inches='tight')