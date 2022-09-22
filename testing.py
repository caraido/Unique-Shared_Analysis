import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import ortho_group
from unique_shared_analysis import UniqueSharedAnalysis
import matplotlib.pyplot as plt
import pandas as pd

#Function that creates a sine wave for a given amount of time (T),
#where the number of cycles (c) occurs during that time
def create_sine(T,c):
    tau=T/(2*np.pi)/c
    return np.sin(np.arange(0,T)/tau)


record_duration = 5  # second
sampling_rate = 100  # Hz
num_neuron = 60  # for each trial type (what if the number of trial is different for each type?)
R = 6  # hidden dimension


# generate latent matrix V
V = ortho_group.rvs(dim=num_neuron)
V1=V[:, 0:R]
V2=V[:, R:2*R]

# generate non-correlated data
z1 = np.random.normal(loc=0, scale=10e-1, size=(sampling_rate * record_duration, R))
z1[:,1:]=0
z2 = np.random.normal(loc=0, scale=10e-1, size=(sampling_rate * record_duration, R))

# generate correlated data
#####################
###### option 1 #####
#####################
# The desired mean values of the sample.
mu = np.array([0.0]*(R+1))

# The desired covariance matrix.
r = np.random.normal(loc=0, scale=10e-1, size=(R+1, R+1))
r=r.T@r
diag=np.random.normal(loc=1, scale=2*10e-2, size=R+1)
r=r+np.diag(diag)

# Generate correlated random samples.
rng = np.random.default_rng()
zs = rng.multivariate_normal(mu, r, size=sampling_rate * record_duration)
#z1=np.zeros([sampling_rate * record_duration, R])
#z2=np.zeros([sampling_rate * record_duration, R])
#z1[:,0]=zs[:,0]
#z2=zs[:,1:]
#zs1[:,0:4]=zs[:,0:4]
#zs2[:,0:2]=zs[:,5:7]

# reconstruct X1 to be the input
X1= z1@V1.T
X2= z2@V2.T

data=np.array([X1,X2])

########################
###### option 2 ########
########################
np.random.seed(0) # To get the same simulated data
T=400 # Time
N=50 # Number of neurons
R_sim=4 # Number of total dimensions in lowD representations

V = ortho_group.rvs(dim=N)
V=V[:,0:R_sim]
#Create low dimensional space
Z=np.zeros([T,R_sim])
for i in range(R_sim):
    Z[100*i:100*i+100,i]=create_sine(100,i+1)
#Create high-dimensional neural activity
X0=Z@V[:R_sim,:] #Project into high-dimensional space
X0=X0+0.01*np.random.randn(X0.shape[0],X0.shape[1]) #Add noise
# this
Z1=np.copy(Z[:200,:2])
#Z1=np.concatenate([Z1,np.zeros([200,2])],axis=0)
Z2=np.copy(Z[200:,2:])
#Z2=np.concatenate([np.zeros([200,2]),Z2],axis=0)

# or this
#Z1=np.array([np.copy(Z[:200,0]),np.copy(Z[200:,2])]).T
#Z2=np.array([np.copy(Z[:200,1]),np.copy(Z[200:,3])]).T

V1=V[:,:2]
V2=V[:,2:]
X1=Z1@V1.T
X1=X1+0.01*np.random.randn(X1.shape[0],X1.shape[1])
X2=Z2@V2.T
X2=X2+0.01*np.random.randn(X2.shape[0],X2.shape[1])
X_concat=np.vstack([X1,X2])

##############
# method one #
##############
R_est=int(R_sim)
USA=UniqueSharedAnalysis(hidden_size=R_est)
USA.initialize(np.array([X1,X2]))
USA.fit(np.array([X1,X2]),n_epochs=500)

V1_hat=USA.transition_mat.V1[-1]
V2_hat=USA.transition_mat.V2[-1]
Vs_hat=USA.transition_mat.Vs[-1]

##############
# method two #
##############
pca=PCA(n_components=R_est)
pca.fit(X_concat)
W=pca.components_

z1_dpca=X1@W.T
z2_dpca=X2@W.T
#z_dpca=X_concat@W.T
#z1_dpca=z_dpca[:400,:]
#z2_dpca=z_dpca[400:,:]

# sort the latent by their variances
z1_dpca_var=np.var(z1_dpca,axis=0)
z2_dpca_var=np.var(z2_dpca,axis=0)
z1_dpca=[list(x) for x in z1_dpca.T]
z1_var_index=list(np.argsort(z1_dpca_var))
z1_dpca=np.array(pd.DataFrame(data=[z1_dpca,z1_dpca_var]).T.sort_values(by=1,ascending=False)[0].tolist()).T

z2_dpca=[list(x) for x in z2_dpca.T]
z2_var_index=list(np.argsort(z2_dpca_var))
z2_dpca=np.array(pd.DataFrame(data=[z2_dpca,z2_dpca_var]).T.sort_values(by=1,ascending=False)[0].tolist()).T

z1_dpca_var=np.var(z1_dpca,axis=0)
z2_dpca_var=np.var(z2_dpca,axis=0)

# original latent variance
z1_var=np.var(z1,axis=0)
z2_var=np.var(z2,axis=0)

z1_usa=USA.latent_variables.x1v1[-1]
z1_usa_var=np.var(z1_usa,axis=0)
z1_cov=z1_usa.T@z1_usa

z2_usa=USA.latent_variables.x2v2[-1]
z2_usa_var=np.var(z2_usa,axis=0)
z2_cov=z2_usa.T@z2_usa

ylim_dpca=[np.minimum(np.min(z1_dpca),np.min(z2_dpca)),np.maximum(np.max(z1_dpca),np.max(z2_dpca))]
ylim_usa=[np.minimum(np.min(z1_usa),np.min(z2_usa)),np.maximum(np.max(z1_usa),np.max(z2_usa))]

plt.figure()
plt.plot(USA.losses)
plt.show()

plt.figure()
plt.plot(np.mean(X1,axis=1))
plt.plot(np.mean(X2,axis=1))
plt.title('X1(blue) and X2(orange)')
plt.show()

fig,ax=plt.subplots(8,3,figsize=(10,12))
ax[0][0].plot(Z1[:,0])
ax[0][0].set_title("original")
ax[0][0].set_xticks([])
ax[0][0].set_xlabel(f'var={np.var(Z1,axis=0)[0]:0.2f}')
ax[0][0].set_ylabel('z1')
ax[1][0].plot(Z1[:,1])
ax[1][0].set_xlabel(f'var={np.var(Z1,axis=0)[1]:0.2f}')
ax[1][0].set_xticks([])
ax[1][0].set_ylabel('z1')
ax[4][0].plot(Z2[:,0],c='orange')
ax[4][0].set_xlabel(f'var={np.var(Z2,axis=0)[0]:0.2f}')
ax[4][0].set_xticks([])
ax[4][0].set_ylabel('z2')
ax[5][0].plot(Z2[:,1],c='orange')
ax[5][0].set_xlabel(f'var={np.var(Z2,axis=0)[1]:0.2f}')
ax[5][0].set_xticks([])
ax[5][0].set_ylabel('z2')


ax[0][1].plot(z1_dpca[:,0])
ax[0][1].set_title("dPCA")
ax[0][1].set_xlabel(f'var={z1_dpca_var[0]:0.2f}')
ax[0][1].set_ylim(ylim_dpca)
ax[0][1].set_xticks([])
ax[1][1].plot(z1_dpca[:,1])
ax[1][1].set_xlabel(f'var={z1_dpca_var[1]:0.2f}')
ax[1][1].set_ylim(ylim_dpca)
ax[1][1].set_xticks([])
ax[2][1].plot(z1_dpca[:,2])
ax[2][1].set_xlabel(f'var={z1_dpca_var[2]:0.2f}')
ax[2][1].set_ylim(ylim_dpca)
ax[2][1].set_xticks([])
ax[3][1].plot(z1_dpca[:,3])
ax[3][1].set_xlabel(f'var={z1_dpca_var[3]:0.2f}')
ax[3][1].set_ylim(ylim_dpca)
ax[3][1].set_xticks([])
ax[4][1].plot(z2_dpca[:,0],c='orange')
ax[4][1].set_xlabel(f'var={z2_dpca_var[0]:0.2f}')
ax[4][1].set_ylim(ylim_dpca)
ax[4][1].set_xticks([])
ax[5][1].plot(z2_dpca[:,1],c='orange')
ax[5][1].set_xlabel(f'var={z2_dpca_var[1]:0.2f}')
ax[5][1].set_ylim(ylim_dpca)
ax[5][1].set_xticks([])
ax[6][1].plot(z2_dpca[:,2],c='orange')
ax[6][1].set_xlabel(f'var={z2_dpca_var[2]:0.2f}')
ax[6][1].set_ylim(ylim_dpca)
ax[6][1].set_xticks([])
ax[7][1].plot(z2_dpca[:,3],c='orange')
ax[7][1].set_xlabel(f'var={z2_dpca_var[3]:0.2f}')
ax[7][1].set_ylim(ylim_dpca)


ax[0][2].plot(z1_usa[:,0])
ax[0][2].set_title("USA")
ax[0][2].set_xlabel(f'var={z1_usa_var[0]:0.2f}')
ax[0][2].set_ylim(ylim_usa)
ax[0][2].set_xticks([])
ax[1][2].plot(z1_usa[:,1])
ax[1][2].set_xlabel(f'var={z1_usa_var[1]:0.2f}')
ax[1][2].set_ylim(ylim_usa)
ax[1][2].set_xticks([])
ax[2][2].plot(z1_usa[:,2])
ax[2][2].set_xlabel(f'var={z1_usa_var[2]:0.2f}')
ax[2][2].set_ylim(ylim_usa)
ax[2][2].set_xticks([])
ax[3][2].plot(z1_usa[:,3])
ax[3][2].set_xlabel(f'var={z1_usa_var[3]:0.2f}')
ax[3][2].set_ylim(ylim_usa)
ax[3][2].set_xticks([])
ax[4][2].plot(z2_usa[:,0],c='orange')
ax[4][2].set_xlabel(f'var={z2_usa_var[0]:0.2f}')
ax[4][2].set_ylim(ylim_usa)
ax[4][2].set_xticks([])
ax[5][2].plot(z2_usa[:,1],c='orange')
ax[5][2].set_xlabel(f'var={z2_usa_var[1]:0.2f}')
ax[5][2].set_ylim(ylim_usa)
ax[5][2].set_xticks([])
ax[6][2].plot(z2_usa[:,2],c='orange')
ax[6][2].set_xlabel(f'var={z2_usa_var[2]:0.2f}')
ax[6][2].set_ylim(ylim_usa)
ax[6][2].set_xticks([])
ax[7][2].plot(z2_usa[:,3],c='orange')
ax[7][2].set_xlabel(f'var={z2_usa_var[3]:0.2f}')
ax[7][2].set_ylim(ylim_usa)

fig.delaxes(ax[2][0])
fig.delaxes(ax[3][0])
fig.delaxes(ax[6][0])
fig.delaxes(ax[7][0])

plt.tight_layout()
plt.show()

# plot individual loss
plt.figure()
plt.plot(np.array(USA.latent_variables.var_x1v1) + 0.1*np.random.randn(USA._n_epochs+1)/20) # add some jitter
plt.plot(np.array(USA.latent_variables.var_x1vs)+ 0.1*np.random.randn(USA._n_epochs+1)/20)# add some jitter
plt.plot(np.array(USA.latent_variables.var_x1v2)+ 0.1*np.random.randn(USA._n_epochs+1)/20)# add some jitter
plt.plot(np.array(USA.latent_variables.var_x2v1)+ 0.1*np.random.randn(USA._n_epochs+1)/20)# add some jitter
plt.plot(np.array(USA.latent_variables.var_x2vs)+ 0.1*np.random.randn(USA._n_epochs+1)/20)# add some jitter
plt.plot(np.array(USA.latent_variables.var_x2v2)+ 0.1*np.random.randn(USA._n_epochs+1)/20)# add some jitter
#plt.plot(reconstructed_error_X1)
#plt.plot(reconstructed_error_X2)
plt.legend(['x1v1','x1vs','x1v2','x2v1','x2vs','x2v2'])
plt.title('individual estimated losses-- X1, X2')
plt.ylabel('var(estimated)')
plt.xlabel('epoch')
plt.show()





