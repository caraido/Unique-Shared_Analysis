
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import io
import torch
from model import fit_comp
from utils_visualization import *
import warnings
from sklearn.decomposition import PCA, TruncatedSVD

def load_data(path):
    data=io.loadmat(path)

    targ_array = data['targ_array']  # Array of preparatory data (num_cond x num_neur x num_time_bins)
    move_array = data['move_array']  # Array of movement data (num_cond x num_neur x num_time_bins)
    interp_array = data['interp_array']  # Array of whole time data (num_cond x num_neur x num_time_bins)
    return targ_array, move_array, interp_array

def preprocess_data(targ_array,move_array,all_array):
    # this is the step mentioned in the paper
    # preprocess the data
    # "Soft normalize" each neuron's activity by dividing it by (its maximum + 5 spks/sec)

    # Get maximum activity over two epochs
    m1_max_tgt = np.max(np.max(targ_array, axis=2), axis=0)
    m1_max_move = np.max(np.max(move_array, axis=2), axis=0)
    m1_max = np.max(np.concatenate((m1_max_tgt[:, None], m1_max_move[:, None]), axis=1), axis=1)

    m1_tgt_align_norm = targ_array / (m1_max[None, :, None] + .05)  # bins are 10ms so this corresponds to 5 spks/sec
    m1_move_align_norm = move_array / (m1_max[None, :, None] + .05)
    m1_all_align_norm=all_array/(np.max(np.max(move_array, axis=2), axis=0)[None,:,None]+.05)

    # Subtract the mean activity across conditions (what they do in the paper)
    m1_tgt_preproc = m1_tgt_align_norm - np.mean(m1_tgt_align_norm, axis=0)[None, :, :]
    m1_move_preproc = m1_move_align_norm - np.mean(m1_move_align_norm, axis=0)[None, :, :]
    m1_all_preproc=m1_all_align_norm-np.mean(m1_all_align_norm,axis=0)[None,:,:]

    # Reshape to concatenate the conditions, so its shape (num_neurons x  time*num_conditions)
    m1_tgt_concat = m1_tgt_preproc.swapaxes(0, 1).reshape(
        [m1_tgt_preproc.shape[1], m1_tgt_preproc.shape[0] * m1_tgt_preproc.shape[2]])
    m1_move_concat = m1_move_preproc.swapaxes(0, 1).reshape(
        [m1_move_preproc.shape[1], m1_move_preproc.shape[0] * m1_move_preproc.shape[2]])
    m1_all_concat = m1_all_preproc.swapaxes(0, 1).reshape(
        [m1_all_preproc.shape[1], m1_all_preproc.shape[0] * m1_all_preproc.shape[2]])

    # adjust the variance of the data
    m1_move_concat = m1_move_concat / np.std(m1_move_concat)
    m1_tgt_concat = m1_tgt_concat / np.std(m1_tgt_concat)
    m1_all_concat=m1_all_concat/np.std(m1_all_concat)

    return m1_tgt_concat,m1_move_concat,m1_all_concat

class LatentVariables:
    def __init__(self,hidden):
        self.x1v1 = [z[0][0].detach().numpy() for z in hidden]
        self.x1vs = [z[0][1].detach().numpy() for z in hidden]
        self.x1v2 = [z[0][2].detach().numpy() for z in hidden]
        self.x2v1 = [z[1][0].detach().numpy() for z in hidden]
        self.x2vs = [z[1][1].detach().numpy() for z in hidden]
        self.x2v2 = [z[1][2].detach().numpy() for z in hidden]

        self.var_x1v1=[np.var(h) for h in self.x1v1]
        self.var_x1vs = [np.var(h) for h in self.x1vs]
        self.var_x1v2 = [np.var(h) for h in self.x1v2]
        self.var_x2v1 = [np.var(h) for h in self.x2v1]
        self.var_x2vs = [np.var(h) for h in self.x2vs]
        self.var_x2v2 = [np.var(h) for h in self.x2v2]


class Reconstructed:
    def __init__(self,reconstructed):
        #((x1v1v1, x1vsvs, x1v2v2), (x2v1v1, x2vsvs, x2v2v2))
        self.X1V1V1=[r[0][0].detach().numpy() for r in reconstructed]
        self.X1VsVs = [r[0][1].detach().numpy() for r in reconstructed]
        self.X1V2V2 = [r[0][2].detach().numpy() for r in reconstructed]
        self.X2V1V1 = [r[1][0].detach().numpy() for r in reconstructed]
        self.X2VsVs = [r[1][1].detach().numpy() for r in reconstructed]
        self.X2V2V2 = [r[1][2].detach().numpy() for r in reconstructed]

        self.X1 = [self.X1VsVs[i]+self.X1V1V1[i] for i in
                                 range(len(reconstructed))]
        self.X2 = [self.X2VsVs[i]+self.X2V2V2[i] for i in
                                 range(len(reconstructed))]

        self.var_X1V1V1 = [np.var(r) for r in self.X1V1V1]
        self.var_X1VsVs = [np.var(r) for r in self.X1VsVs]
        self.var_X1V2V2 = [np.var(r) for r in self.X1V2V2]
        self.var_X2V1V1 = [np.var(r) for r in self.X2V1V1]
        self.var_X2VsVs = [np.var(r) for r in self.X2VsVs]
        self.var_X2V2V2 = [np.var(r) for r in self.X2V2V2]

        self.var_X1=[np.var(r) for r in self.X1]
        self.var_X2 = [np.var(r) for r in self.X2]


class TransitionMat:
    def __init__(self,mat):
        #(V1_weight, Vs_weight, V2_weight)
        self.V1=[v[0].detach().numpy() for v in mat]
        self.Vs = [v[1].detach().numpy() for v in mat]
        self.V2 = [v[2].detach().numpy() for v in mat]


class UniqueSharedAnalysis:

    def __init__(self,hidden_size):
        self._hidden_size=None
        self._raw_data=None
        self._raw_data_size=None
        self._lr=None
        self._n_epochs=None
        self._want_bias=None
        self._want_stats=False

        self.run = False
        self.hidden_size=hidden_size
        self.losses=None
        self.latent_variables=None
        self.reconstructed=None
        self.transition_mat=None
        self.model=None
        self.X1=None
        self.X2=None

    def fit(self, data,lr=0.001, n_epochs=1000,want_bias=False,want_stats=True):
        # check the format of the data
        if self.check_format(data):
            self._raw_data=data
            self.X1=data[0]
            self.X2=data[1]
            self._lr=lr
            self._n_epochs=n_epochs
            self._want_bias=want_bias
            self._want_stats=want_stats

            print("#### before training ###\n")
            print(f"For X1 --- mean: {np.mean(self.X1):.2f}, variance: {np.var(self.X1):.2f}\nFor X2 --- mean: {np.mean(self.X2):.2f}, variance: {np.var(self.X2):.2f}\n")

            loss, hidden, reconstructed, matrices, model = fit_comp(self._raw_data, R=self.hidden_size, lr=self._lr, n_epochs=self._n_epochs,
                                                                    want_bias=self._want_bias)
            self.run=True

            self.losses=loss
            self.latent_variables=LatentVariables(hidden)
            del hidden # to save the memory
            self.reconstructed=Reconstructed(reconstructed)
            del reconstructed # to save the memory
            self.transition_mat=TransitionMat(matrices)
            del matrices # to save the memory
            self.model=model

            if self._want_stats:
                self._calculate_stats()

            print(f"Original total loss:{loss[0]:.2f}. ")
            print('### after training ###')
            print(f"Last epoch total loss:{loss[-1]:.2f}. ")
        else:
            raise Exception("input data should be a 3 dimensional ndarray/tensor with first D=2 .")

    @property
    def hidden_size(self):
        return self._hidden_size

    @hidden_size.setter
    def hidden_size(self, new_hidden_size):
        if self.run:
            self._hidden_size=new_hidden_size
            loss, hidden, reconstructed, matrices, model = fit_comp(self._raw_data, R=self.hidden_size, lr=self._lr, n_epochs=self._n_epochs,
                                                                    want_bias=self._want_bias)
            self.losses = loss
            self.latent_variables = LatentVariables(hidden)
            self.reconstructed = Reconstructed(reconstructed)
            self.transition_mat = TransitionMat(matrices)
            self.model = model
            if self._want_stats:
                self._calculate_stats()
        else:
            self._hidden_size=new_hidden_size

    def check_format(self,data):
        if isinstance(data,np.ndarray):
            size=data.shape
            if size[0]==2 and len(size)==3:
                self.raw_data_size=size
                return True
            else: return False
        elif isinstance(data,torch.Tensor):
            size=data.size()
            if size[0]==2 and len(size)==3:
                self.raw_data_size=size
                data=data.detach().numpy()
                warnings.warn(f"Preferred numpy array data type but got {data.type()}. Now transforming to numpy array")
                return True
            else: return False
        else: return False

    def _calculate_stats(self):
        # input reconstruction
        self.reconstruction_X1=self.reconstructed.X1
        self.reconstruction_X2=self.reconstructed.X2
        # pearson correlation
        self.X1_pearson_corr=[np.corrcoef(self.reconstruction_X1[i].flatten(),self.X1.flatten()) for i in range(self._n_epochs+1)]
        self.X2_pearson_corr=[np.corrcoef(self.reconstruction_X2[i].flatten(),self.X2.flatten()) for i in range(self._n_epochs+1)]

        # reconstruction error
        self.X1_recon_MSE=[np.mean(np.square(self.reconstruction_X1[i]-self.X1)) for i in range(self._n_epochs+1)]
        self.X2_recon_MSE = [np.mean(np.square(self.reconstruction_X2[i] - self.X2)) for i in range(self._n_epochs+1)]

    @property
    def n_epochs(self):
        return self._n_epochs


if __name__=='__main__':
    from jPCA import jPCA
    from jPCA.util import plot_projections
    # load the data first
    load_folder = '/Users/tianhaolei/PycharmProjects/mini_rotation_josh/'
    path=load_folder + 'monkey_n_avgs'
    targ_array,move_array,all_array=load_data(path)
    data_shape=targ_array.shape #(num_cond x num_neuron x num_time_bins)
    all_shape=all_array.shape

    m1_tgt_concat,m1_move_concat,m1_all_concat=preprocess_data(targ_array,move_array,all_array)

    # check stats
    plt.figure()
    plt.boxplot([m1_tgt_concat.flatten(),m1_move_concat.flatten()],sym='')
    plt.xticks(ticks=[1,2],labels=['preparatory', 'movement'])
    print(f'For preparatory data --- mean : {np.mean(m1_tgt_concat.flatten())}, variance: {np.var(m1_tgt_concat.flatten())}')
    print(f'For movement data --- mean : {np.mean(m1_move_concat.flatten())}, variance: {np.var(m1_move_concat.flatten())}')
    # use SVD as initialization (not implement here )

    # run the model with the initialization
    X=np.array([m1_tgt_concat.T,m1_move_concat.T])
    hidden_size=10
    USA=UniqueSharedAnalysis(hidden_size=hidden_size)
    USA.fit(X)

    all2V1=m1_all_concat.T@USA.transition_mat.V1[-1]
    all2V2=m1_all_concat.T@USA.transition_mat.V2[-1]
    all2Vs=m1_all_concat.T@USA.transition_mat.Vs[-1]

    var_all2V1=np.var(all2V1)
    var_all2V2=np.var(all2V2)
    var_all2Vs=np.var(all2Vs)

    print(f"variance of prep+move project into V1(unique to preparation): {var_all2V1:.2f}")
    print(f"variance of prep+move project into Vs(shared space): {var_all2Vs:.2f}")
    print(f"variance of prep+move project into V2(unique to movement): {var_all2V2:.2f}")
    # inspect the last epoch
    epoch=-1
    group_scatter([m1_tgt_concat.T.flatten(),m1_move_concat.T.flatten()], [USA.reconstruction_X1[epoch].flatten(), USA.reconstruction_X2[epoch].flatten()], ['X1', 'X2'],
                  title="reconstructed data comparison")

    # plot individual loss
    plt.figure()
    plt.plot(np.array(USA.latent_variables.var_x1v1) + np.random.randn(USA._n_epochs+1)/10) # add some jitter
    plt.plot(np.array(USA.latent_variables.var_x1vs)+ np.random.randn(USA._n_epochs+1)/10)# add some jitter
    plt.plot(np.array(USA.latent_variables.var_x1v2)+ np.random.randn(USA._n_epochs+1)/10)# add some jitter
    plt.plot(np.array(USA.latent_variables.var_x2v1)+ np.random.randn(USA._n_epochs+1)/10)# add some jitter
    plt.plot(np.array(USA.latent_variables.var_x2vs)+ np.random.randn(USA._n_epochs+1)/10)# add some jitter
    plt.plot(np.array(USA.latent_variables.var_x2v2)+ np.random.randn(USA._n_epochs+1)/10)# add some jitter
    #plt.plot(reconstructed_error_X1)
    #plt.plot(reconstructed_error_X2)

    plt.legend(['x1v1','x1vs','x1v2','x2v1','x2vs','x2v2'])
    plt.title('individual estimated losses-- X1:preparatory, X2: movement')
    plt.ylabel('var(estimated)')
    plt.xlabel('epoch')
    plt.show()

    plt.figure()
    plt.plot(USA.losses)
    plt.legend(['training loss'])

    plt.figure()
    plt.plot(USA.X1_recon_MSE)
    plt.plot(USA.X2_recon_MSE)
    plt.legend(['preparatory', 'movement'])
    plt.title('reconstruction MSE')
    plt.show()

    # plot the data
    c_size=0.01
    a_size=0.02
    # reshape the latent variables
    x1v1=USA.latent_variables.x1v1[epoch].reshape(data_shape[0],data_shape[2],hidden_size)
    x1vs = USA.latent_variables.x1vs[epoch].reshape(data_shape[0], data_shape[2], hidden_size)
    x2vs = USA.latent_variables.x2vs[epoch].reshape(data_shape[0], data_shape[2], hidden_size)
    x2v2 = USA.latent_variables.x2v2[epoch].reshape(data_shape[0], data_shape[2], hidden_size)

    x1v1v1=USA.reconstructed.X1V1V1[epoch].reshape(data_shape[0],data_shape[2],data_shape[1])
    x1vsvs = USA.reconstructed.X1VsVs[epoch].reshape(data_shape[0], data_shape[2], data_shape[1])
    x2vsvs = USA.reconstructed.X2VsVs[epoch].reshape(data_shape[0], data_shape[2], data_shape[1])
    x2v2v2 = USA.reconstructed.X2V2V2[epoch].reshape(data_shape[0], data_shape[2], data_shape[1])

    all2V1=all2V1.reshape(all_shape[0],all_shape[2],hidden_size)
    all2Vs=all2Vs.reshape(all_shape[0],all_shape[2],hidden_size)
    all2V2=all2V2.reshape(all_shape[0],all_shape[2],hidden_size)

    print(f"var(x1v1): variance of unique component of movement preparation: {USA.latent_variables.var_x1v1[-1]:.2f})")
    print(f"var(x1vs): variance of shared component of movement preparation: {USA.latent_variables.var_x1vs[-1]:.2f})")
    print(f"var(x2v2): variance of unique component of movement execution: {USA.latent_variables.var_x2v2[-1]:.2f})")
    print(f"var(x2vs): variance of shared component of movement execution: {USA.latent_variables.var_x2vs[-1]:.2f})")


    # jPCA analysis for latent variables
    x1v1_all = [x for x in x1v1]
    x1vs_all =[x for x in x1vs]
    x2v2_all = [x for x in x2v2]
    x2vs_all =[x for x in x2vs]
    jpca = jPCA.JPCA(num_jpcs=2)
    (x1v1_projected,
     x1v1_cps,
     x1v1_full_data_var,
     x1v1_pca_var_capt,
     x1v1_jpca_var_capt) = jpca.fit(x1v1_all)
    (x1vs_projected,
     x1vs_cps,
     x1vs_full_data_var,
     x1vs_pca_var_capt,
     x1vs_jpca_var_capt) = jpca.fit(x1vs_all)
    (x2vs_projected,
     x2vs_cps,
     x2vs_full_data_var,
     x2vs_pca_var_capt,
     x2vs_jpca_var_capt) = jpca.fit(x2vs_all)
    (x2v2_projected,
     x2v2_cps,
     x2v2_full_data_var,
     x2v2_pca_var_capt,
     x2v2_jpca_var_capt) = jpca.fit(x2v2_all)


    fig,ax=plt.subplots(2,2,figsize=(8,8))

    plot_projections(x1v1_projected,axis=ax[0][0],x_idx=0, y_idx=1,arrow_size=a_size,circle_size=c_size)
    ax[0][0].set_title('x1v1')
    ax[0][0].set_xlabel('jPC 1')
    ax[0][0].set_ylabel('jPC 2')

    plot_projections(x1vs_projected,axis=ax[0][1],x_idx=0, y_idx=1,arrow_size=a_size,circle_size=c_size)
    ax[0][1].set_title('x1vs')
    ax[0][1].set_xlabel('jPC 1')
    ax[0][1].set_ylabel('jPC 2')

    plot_projections(x2v2_projected,axis=ax[1][0],x_idx=0, y_idx=1,arrow_size=a_size,circle_size=c_size)
    ax[1][0].set_title('x2v2')
    ax[1][0].set_xlabel('jPC 1')
    ax[1][0].set_ylabel('jPC 2')

    plot_projections(x2vs_projected,axis=ax[1][1],x_idx=0, y_idx=1,arrow_size=a_size,circle_size=c_size)
    ax[1][1].set_title('x2vs')
    ax[1][1].set_xlabel('jPC 1')
    ax[1][1].set_ylabel('jPC 2')
    fig.suptitle('latent variables jPCA analysis')

    plt.tight_layout()
    plt.show()

    # jPCA analysis for reconstruction
    x1v1v1_all = [x for x in x1v1v1]
    x1vsvs_all = [x for x in x1vsvs]
    x2v2v2_all = [x for x in x2v2v2]
    x2vsvs_all = [x for x in x2vsvs]
    jpca = jPCA.JPCA(num_jpcs=2)
    (x1v1v1_projected,
     _,_,_,_) = jpca.fit(x1v1v1_all)
    (x1vsvs_projected,
     _,_,_,_) = jpca.fit(x1vsvs_all)
    (x2vsvs_projected,
     _,_,_,_) = jpca.fit(x2vsvs_all)
    (x2v2v2_projected,
     _,_,_,_) = jpca.fit(x2v2v2_all)

    fig2, ax = plt.subplots(2, 2,figsize=(8,8))

    plot_projections(x1v1v1_projected, axis=ax[0][0], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    ax[0][0].set_title('x1v1v1')
    ax[0][0].set_xlabel('jPC 1')
    ax[0][0].set_ylabel('jPC 2')

    plot_projections(x1vsvs_projected, axis=ax[0][1], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    ax[0][1].set_title('x1vsvs')
    ax[0][1].set_xlabel('jPC 1')
    ax[0][1].set_ylabel('jPC 2')

    plot_projections(x2v2v2_projected, axis=ax[1][0], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    ax[1][0].set_title('x2v2v2')
    ax[1][0].set_xlabel('jPC 1')
    ax[1][0].set_ylabel('jPC 2')

    plot_projections(x2vsvs_projected, axis=ax[1][1], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    ax[1][1].set_title('x2vsvs')
    ax[1][1].set_xlabel('jPC 1')
    ax[1][1].set_ylabel('jPC 2')

    fig2.suptitle('reconstruction jPCA analysis')
    plt.tight_layout()
    plt.show()

    # jPCA analysis for reconstruction
    all2v1_all = [x for x in all2V1]
    all2vs_all = [x for x in all2Vs]
    all2v2_all = [x for x in all2V2]
    jpca = jPCA.JPCA(num_jpcs=2)
    (all2v1_projected,
     _,_, _, _) = jpca.fit(all2v1_all)
    (all2vs_projected,
     _,_, _, _) = jpca.fit(all2vs_all)
    (all2v2_projected,
     _,_, _, _) = jpca.fit(all2v2_all)

    fig3, ax = plt.subplots(2, 2,figsize=(8,8))

    plot_projections(all2v1_projected, axis=ax[0][0], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    ax[0][0].set_title('all to V1')
    ax[0][0].set_xlabel('jPC 1')
    ax[0][0].set_ylabel('jPC 2')

    plot_projections(all2vs_projected, axis=ax[0][1], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    ax[0][1].set_title('all to Vs')
    ax[0][1].set_xlabel('jPC 1')
    ax[0][1].set_ylabel('jPC 2')

    plot_projections(all2v2_projected, axis=ax[1][0], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    ax[1][0].set_title('all to V2')
    ax[1][0].set_xlabel('jPC 1')
    ax[1][0].set_ylabel('jPC 2')

    fig3.suptitle('all data projection jPCA analysis')
    plt.tight_layout()
    plt.show()


