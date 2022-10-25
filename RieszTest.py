import numpy as np
import matplotlib.pyplot as plt
import torch
import Riesz as R
import os
import glob
import imageio.v2 as imageio
#from pykeops.torch import kernel_product, Genred
#from pykeops.torch.kernel_product.formula import *

use_cuda = 0
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32
use_keops=False
import Riesz as R
s=2
RS=R.RieszSearcher(s)

d=2 
N=4
max_iter=401
epsilon= .000001
alpha=1/((d*N)**2) #This is just a guess for the learning rate
cnf= np.random.rand(d+1,N)-.5
cnf = R.proj(cnf)
print(cnf)
x = torch.from_numpy(cnf).requires_grad_()

RS.plot(x)
x = torch.from_numpy(cnf).to(dtype=torchdtype, device=torchdeviceId)
new_x,energy,dic,enr_dic=RS.pgd(x,max_iter,epsilon,alpha,display=10)

fdir = "out2"
filenames = sorted(glob.glob(os.path.join(fdir, "out_*.png")))
gif_name = "testing"

print('creating gif\n')
with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
ax.plot(enr_dic)
plt.show

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
ax.set_yscale('log')
ax.plot((enr_dic[0:-1]-enr_dic[1:])/enr_dic[-1])
plt.show()

RS.plot(new_x)