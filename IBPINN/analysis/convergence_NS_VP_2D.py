#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:49:40 2022

@author: chaztikov
"""
import deepxde
import deepxde as dde
import numpy as np
# import deepxde.backend
# from deepxde.backend import tf
import scipy;from scipy.interpolate import griddata
# import deepxde.backend.tensorflow as tf
import matplotlib.pyplot as plt

# from tensorflow.keras.backend import set_floatx
# set_floatx("float64")

plt.ioff()

# deepxde.config.set_random_seed(1234)
# import sklearn
# scaler=sklearn.preprocessing.StandardScaler();scaler

# from deepxde.metrics import l2_relative_error,mean_absolute_percentage_error,mean_l2_relative_error,nanl2_relative_error
# from deepxde.callbacks import EarlyStopping,FirstDerivative,ModelCheckpoint,MovieDumper,OperatorPredictor,Timer#
import os,re,sys,time

data_directory = ['/media/chaztikov/SlowStorage/thesis/PINNs/IBPINN/projects/',
                  '/home/chaztikov/git/PINNs/main/examples/IBPINN/projects/',
                  '/media/chaztikov/SlowStorage/thesis/PINNs/IBPINN/projects/'][0]
data_directory += ['FOSLS/steady_navierstokes_newtonian/newtonian_channel_laminar_2D/saved/navierstokes_newtonian_channel_2_0_stationary/NS_VVT_FOSLS_2D/hagenpoiseuille_channel_2D/',
                   'FOSLS/convergence/num_collocation/saved/navierstokes_newtonian_channel_2_0_stationary/NS_VVT_FOSLS_2D/hagenpoiseuille_channel_1D/',
                   'FOSLS/VVT/BC0/saved/navierstokes_newtonian_channel_2_0_stationary/NS_2D/hagenpoiseuille_channel_2D/'][-1]
data_directory  = '/media/chaztikov/SlowStorage/thesis/PINNs/IBPINN/projects/FOSLS/steady_navierstokes_newtonian/newtonian_channel_laminar_2D/saved/navierstokes_newtonian_channel_2_0_stationary/NS_VVT_FOSLS_2D/hagenpoiseuille_channel_2D/'
data_directory = '/media/chaztikov/SlowStorage/thesis/PINNs/IBPINN/projects/FOSLS/VVT/BC0/saved/navierstokes_newtonian_channel_2_0_stationary/NS_VVT_FOSLS_2D/hagenpoiseuille_channel_2D/'
data_directory = '/media/chaztikov/SlowStorage/thesis/PINNs/IBPINN/projects/FOSLS/VVP/BC0/saved/navierstokes_newtonian_channel_2_0_stationary/NS_VVP_FOSLS_2D/hagenpoiseuille_channel_2D/'
data_directory = '/media/chaztikov/SlowStorage/thesis/PINNs/IBPINN/projects/FOSLS/VP/BC0/saved/navierstokes_newtonian_channel_2_0_stationary/NS_2D/hagenpoiseuille_channel_2D/'

results_directory=data_directory+'/results/'+str(int(time.time()))+'/'
os.system('mkdir -p '+results_directory)

def get_one_data():
    files = os.listdir(data_directory)
    # files = [f for f in files if f.startswith('PFNN')]
    files = [f for f in files if f.startswith('FNN')]
    
    dname0 = files[0] + '/'
    dnames = os.listdir(data_directory + dname0)
    dname1 = dnames[0]
    files1 = os.listdir(data_directory + dname0 + dname1)
    files2 = [f for f in files1 if f.endswith('.dat') and f.startswith('trainstep_')]
    trainstep = max(list(map(lambda s: int(''.join(re.findall('\d',s))) ,files2)))
    
    testfiles2 = [f for f in files1 if f.count('.dat') and f.count('test') and f.count(str(trainstep ))][0] 
    trainfiles2 = [f for f in files1 if f.count('.dat') and f.count('train') and f.count(str(trainstep ))][0]
    lossfiles2 = [f for f in files1 if f.count('.dat') and f.count('loss') and f.count(str(trainstep ))][0]
    
    fname = data_directory + dname0 + dname1 + '/'+lossfiles2
    data = scipy.genfromtxt(fname)
    return data


def get_all_data():
    files = os.listdir(data_directory)
    print(files)
    # files0 = [f for f in files if f.startswith('PFNN')]
    files0 = [f for f in files if f.startswith('FNN')]
    print(files0)
    datas = dict()
    for files in files0:
        try:
            dname0 = files + '/'
            print(dname0)
            dnames = os.listdir(data_directory + dname0)
            dname1 = dnames[0]
            for dname1 in dnames:
                dname2 = data_directory + dname0 + dname1+'/'
                files1 = os.listdir(dname2)
                # print(files1)
                files2 = [f for f in files1 if f.endswith('.dat')]# or f.startswith('trainstep_')]
                files2
                trainstep = np.max(list(map(lambda s: int(''.join(re.findall('\d',s))) ,files2)))
                print(trainstep)
                testfiles = [f for f in files1 if f.count('.dat') and f.count('test') and f.count(str(trainstep ))][0] 
                trainfiles = [f for f in files1 if f.count('.dat') and f.count('train') and f.count(str(trainstep ))][0]
                lossfiles = [f for f in files1 if f.count('.dat') and f.count('loss') and f.count(str(trainstep ))][0]
                
                data = dict()
                keys = ['test','train','loss']
                for key,files3 in zip(keys,[testfiles,trainfiles,lossfiles]):
                    fname = dname2 + files3
                    # print(fname)
                    data[key] = scipy.genfromtxt(fname)
                    # print(data)
                    # data = np.stack()
                    # print(data)
                    datas[key+'_'+files] = data[key]
                # datas['error'+'_'+files] = data[;tes]
        except Exception:
            1
    return datas
datas = get_all_data()
# test_datas
test_keys = [k for k in datas.keys() if k.startswith('test')]
test_datas = [datas[k] for k in datas.keys() if k.startswith('test')]
test_nobservation = [int(tk.split('_')[-1])for tk in test_keys]
test_nboundary = [int(tk.split('_')[-2])for tk in test_keys]
# plt.figure()
# for i in range(len(test_datas)):
#     xx,yy = test_nobservation[i], test_nboundary[i]
#     xx,yy = [xx],[yy]
#     ue,up = np.split(test_datas[i][:,2:],2,1)
#     zz = ((ue.T[0]-up.T[0])**2)
#     # zz /= ((ue.T[0])**2)
#     zz /= np.max((ue.T[0])**2)
#     zz =dde.metrics.mean_squared_error(ue.T[0],up.T[0])
#     zz=[zz]
#     plt.plot(xx,zz);
#     # plt.colorbar();
#     plt.title(test_keys[i])
#     # plt.show();
#     print(test_keys[i])
#     print( ((ue.T[0]-up.T[0])**2).sum() / ((ue.T[0]-0 * up.T[0])**2).sum())
#     print( dde.metrics.mean_squared_error(ue.T[0],up.T[0]) )
#     print( dde.metrics.nanl2_relative_error(ue.T[0],up.T[0]) )
# plt.show();
plt.figure()
for i in range(len(test_datas)):
    # xy,ue,up = test_datas[i]
    xx,yy = np.split(test_datas[i][:,:2],2,1)
    ue,up = np.split(test_datas[i][:,2:],2,1)
    # plt.tricontourf(xx[:,0],yy[:,0],ue.T[0]);plt.show()
    # plt.tricontourf(xx[:,0],yy[:,0],up.T[0]);plt.show()
    zz = ((ue.T[0]-up.T[0])**2)
    # zz /= ((ue.T[0])**2)
    zz /= np.max((ue.T[0])**2)
    plt.tricontourf(xx[:,0],yy[:,0],zz);plt.colorbar();
    plt.title(test_keys[i])
    plt.show();
    print(test_keys[i])
    print( ((ue.T[0]-up.T[0])**2).sum() / ((ue.T[0]-0 * up.T[0])**2).sum())
    # print( dde.metrics.mean_squared_error(ue.T[0],up.T[0]) )
    # print( dde.metrics.nanl2_relative_error(ue.T[0],up.T[0]) )
    



[re.findall('\d',s)[7:] for s in datas.keys()]
loss_keys = [k for k in datas.keys() if k.startswith('loss')]
loss_datas = [datas[k] for k in datas.keys() if k.startswith('loss')]
endloss_datas = [datas[k][-1] for k in datas.keys() if k.startswith('loss')]
endloss_datas = np.vstack(endloss_datas)
endloss_datas.shape

mhyperparameters = np.stack([list(map(int,k.split('_')[4:])) for k in datas.keys() if k.startswith('loss')])
# mhyperparameters = np.stack([list(map(int,k.split('_')[mprefixlen:])) for k in datas.keys() if k.startswith('loss')])
mhyperparameters
nnwidth,nndepth,nnsize = mhyperparameters.T[0] , mhyperparameters.T[1], mhyperparameters.T[0] * mhyperparameters.T[1] 
num_collocation = mhyperparameters.T[3]
num_boundary = mhyperparameters.T[4]
num_observation = mhyperparameters.T[5]
# num_observation
import pandas as pd

mdict = dict()
mdict['loss'] = endloss_datas.T[0]
mdict['nnwidth'] = nnwidth 
mdict['nndepth'] = nndepth 
mdict['nnsize'] = nnsize 
mdict['num_collocation'] = num_collocation 
mdict['num_boundary'] = num_boundary 
mdict['num_observation'] = num_observation 
mdf = pd.DataFrame(mdict)
mdf.plot.scatter('nnwidth','loss');plt.show()
mdf.plot.scatter('nndepth','loss');plt.show()
mdf.plot.scatter('nnsize','loss');plt.show()

# nnwidth
# plt.figure()
# for nc in np.unique(num_collocation):
#     idx = np.where(num_collocation==nc)
#     xx = nnwidth[idx]
#     # xx = nndepth[idx]
#     # xx = nnsize[idx]
#     yy = endloss_datas.T[1][idx]
#     plt.plot(xx ,yy,'o');
# plt.show()

# for ii in range(len(endloss_datas.T)):
#     plt.figure()
#     for nc in np.unique(nnsize):
#         idx = np.where(nnsize==nc)
#         xx = num_collocation[idx]
#         # xx = nndepth[idx]
#         # xx = nnsize[idx]
#         yy = endloss_datas.T[ii][idx]
#         plt.plot(xx ,yy,'.-',label=str(nc));
#     plt.legend()
#     plt.show()
 
# for ii in range(len(endloss_datas.T)):
#     plt.figure()
#     for nc in np.unique(nndepth):
#         idx = np.where(nndepth==nc)
#         xx = num_collocation[idx]
#         # xx = nndepth[idx]
#         # xx = nnsize[idx]
#         yy = endloss_datas.T[ii][idx]
#         plt.plot(xx ,yy,'.-',label=str(nc));
#     plt.legend()
#     plt.show()

titles=[]
plot_titles=(['BFGS Optimization Iteration'])
# titles = (['Total Loss','Continuity','Momentum (x-component)','Momentum (y-component)','curl($\omega$)-v','Dirichlet BC (u component)','Dirichlet BC (v component)'])
titles = (['Total Loss','Momentum (x-component)','Momentum (y-component)','Continuity','Dirichlet BC (u component)','Dirichlet BC (v component)'])
plot_titles.extend(titles)
plot_titles.extend(titles)
# plot_titles.extend(['Total Loss','curl($\omega$)+v','div(v)','Momentum (x-component)','Momentum (y-component)','curl(r)','Divergence of Momentum','Dirichlet BC (u component)','Dirichlet BC (v component)'])
plot_titles.extend(['Metric (L2 Error)'])      
# plot_titles.extend(plot_titles)
len(plot_titles )
plot_titles
lenendloss_datas = len(endloss_datas.T)
lenendloss_datas 
endloss_datas.T[0]-2000
endloss_datas.T[9]
for ii,mtitle in enumerate(plot_titles):
    if ii==0:
        1
    elif ii<10:
        mtitle = mtitle + ', Training Set'
    elif ii==20:
        1
    elif (ii>9)*(ii<20):
        mtitle = mtitle + ', Validation Set'
    # for nb in np.unique(num_boundary):
    #     nnidx=(num_boundary==nb)
    
    # for no in np.unique(num_observation):
    #     nnidx=(num_observation==no)
    plt.figure()
    for nw in np.unique(nnwidth):
        # nnidx=1
        # nnidx = nnidx * (nnwidth==nw)
        for nd in np.unique(nndepth):
            # nnidx = nnidx * (nndepth==nd)
            idx = np.where((nnwidth==nw)*(nndepth==nd))
            xx = num_collocation[idx]
            # xx = nndepth[idx]
            # xx = nnsize[idx]
            yy = endloss_datas.T[ii][idx]
            
            isort = np.argsort(xx)
            xx = xx[isort]
            yy = yy[isort]
            plt.loglog(xx ,yy,'o-',label=(str(nw),str(nd)));
    plt.legend()
    plt.grid()
    plt.title(mtitle)
    plt.xlabel('Number of Collocation Points')
    plt.ylabel('Loss (Mean Squared Error, Training Set)')
    plt.savefig(results_directory+mtitle)#+'/results'+
    plt.show()




plt.figure()
xmin=2020
for xy in loss_datas:
    xx = xy.T[0]
    print(xy.T[1:6][:,-1])
    for yy in xy.T[1:2]:
        plt.semilogy(xx[xx>xmin],yy[xx>xmin],'.-',ms=1);
plt.show()
# print(yy.shape)

xtrain=np.ones([1,10]);ytrain=xtrain;xtest=xtrain;ytest=xtest;net=dde.nn.FNN([10]+[3]+[1],'relu','Glorot uniform');data=dde.data.DataSet(xtrain,ytrain,xtest,ytest);model=dde.Model(data,net);model.compile('L-BFGS');model.train()

model.restore(data_directory)
dde.Model.restore
'''
xx = np.stack(loss_datas).T[0][:]
yvals = np.stack(datas.values()).T[1:12]
xvals = np.stack([list(map(int,k.split('_')[-3:])) for k in datas.keys()]).T
xvals
xvals1 = (xvals[2]/xvals[1])[None,:]
xvals1 = (xvals[2])[None,:]
#xvals[0:1]/(xvals[0:1] + xvals[1]**2 + xvals[2]**2)
plt.figure()
for xx in xvals1:
    for yy in yvals[0:]:
        plt.semilogy(xx,yy,'.-');
plt.show()'''


# data = get_all_data()
# xx = data.T[0]
# yy = data.T[1]
# plt.figure()
# for yy in data.T[1:-12]:
#     plt.semilogy(xx,yy);
# plt.show()

# plt.figure()
# for yy in data.T[-12:]:
#     plt.semilogy(xx,yy);
# plt.show()


# data[-1][1:-12]
