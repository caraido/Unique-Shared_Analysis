
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import io
import torch
from model import UniqueSharedAnalysis
from utils_visualization import *

def load_data(path):
    data=io.loadmat(path)

    targ_array = data['targ_array']  # Array of preparatory data (num_cond x num_neur x num_time_bins)
    move_array = data['move_array']  # Array of movement data (num_cond x num_neur x num_time_bins)
    interp_array = data['interp_array']  # Array of whole time data (num_cond x num_neur x num_time_bins)
    return targ_array, move_array, interp_array

# this is the function to preprocess the data in preparation/movement style
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

# this is the function to preprocess the data in condition1(straight movement)/condition2(curved movement)
def preprocess_data2(all_array):
    straight=all_array[0::3]
    curved1=all_array[1::3]
    curved2=all_array[2::3]

    max_straight = np.max(np.max(straight, axis=2), axis=0)
    max_curved1 = np.max(np.max(curved1, axis=2), axis=0)
    all_max = np.max(np.concatenate((max_straight[:, None], max_curved1[:, None]), axis=1), axis=1)
    straight_align_norm = straight / (all_max[None, :, None] + .05)  # bins are 10ms so this corresponds to 5 spks/sec
    curved1_align_norm = curved1 / (all_max[None, :, None] + .05)

    # Subtract the mean activity across conditions (what they do in the paper)
    straight_preproc = straight_align_norm - np.mean(straight_align_norm, axis=0)[None, :, :]
    curved1_preproc = curved1_align_norm - np.mean(curved1_align_norm, axis=0)[None, :, :]

    # Reshape to concatenate the conditions, so its shape (num_neurons x  time*num_conditions)
    straight_concat = straight_preproc.swapaxes(0, 1).reshape(
        [straight_preproc.shape[1], straight_preproc.shape[0] * straight_preproc.shape[2]])
    curved1_concat = curved1_preproc.swapaxes(0, 1).reshape(
        [curved1_preproc.shape[1], curved1_preproc.shape[0] * curved1_preproc.shape[2]])
    return straight_concat, curved1_concat

def sort_helper(X:np.ndarray, sort_index=0):
    dtype = [('value', np.ndarray), ('index', float)]
    X=[(x,x[sort_index]) for x in X.T]
    X= np.sort(np.array(X, dtype=dtype), order='index')[::-1]
    X=np.array([x[0] for x in X]).T
    return X

def sort_helper_ref(X:np.ndarray, sort_ref, ascending=False):
    # X should be a 2 dimensional data with the first dimension or the last dimension equal to the length of sort reference
    assert X.shape[0]==len(sort_ref) or X.shape[-1]==len(sort_ref)
    dtype = [('value', np.ndarray), ('index', float)]
    if X.shape[-1]==len(sort_ref):
        X=X.T
    X = [(x, sort_ref[i]) for i, x in enumerate(X)]
    if ascending:
        X=np.sort(np.array(X, dtype=dtype), order='index')
    else:
        X= np.sort(np.array(X, dtype=dtype), order='index')[::-1]
    if X.shape[-1]==len(sort_ref):
        X=np.array([x[0] for x in X]).T
    else:
        X=np.array([x[0] for x in X])
    return X

# Using monkey data to demonstrate USA analysis.
if __name__=='__main__':
    from jPCA import jPCA
    from jPCA.util import plot_projections
    import seaborn as sns
    # load the data first
    load_folder = '/Users/tianhaolei/PycharmProjects/mini_rotation_josh/examples/'
    path=load_folder + 'monkey_n_avgs'
    targ_array,move_array,all_array=load_data(path)
    targ_array=np.concatenate([targ_array,move_array],axis=2)
    data_shape=targ_array.shape #(num_cond x num_neuron x num_time_bins)
    all_shape=all_array.shape

    # this function looks into preparation/movement split
    #X1,X2,X_all=preprocess_data(targ_array,move_array,all_array)
    # this function looks into straight/curve split
    X1, X2 = preprocess_data2(targ_array)
    data_shape=list(targ_array.shape)
    data_shape[0]=int(data_shape[0]/3) # this specifically applies on straight/curve split


    # check stats
    plt.figure()
    plt.boxplot([X1.flatten(),X2.flatten()],sym='')
    plt.xticks(ticks=[1,2],labels=['straight', 'curved'])

    # run the model with the initialization
    X=np.array([X1.T,X2.T])
    hidden_size=10
    USA=UniqueSharedAnalysis(hidden_size=hidden_size)
    USA.initialize(X,method='iter')
    USA.fit(X,n_epochs=8)

    #all2V1=X_all.T@USA.transition_mat.V1[-1]
    #all2V2=X_all.T@USA.transition_mat.V2[-1]
    #all2Vs=X_all.T@USA.transition_mat.Vs[-1]

    #var_all2V1=np.var(all2V1)
    #var_all2V2=np.var(all2V2)
    #var_all2Vs=np.var(all2Vs)

    #print(f"variance of prep+move project into V1(unique to preparation): {var_all2V1:.2f}")
    #print(f"variance of prep+move project into Vs(shared space): {var_all2Vs:.2f}")
    #print(f"variance of prep+move project into V2(unique to movement): {var_all2V2:.2f}")
    # inspect the last epoch
    epoch=-1
    group_scatter([X1.T.flatten(),X2.T.flatten()], [USA.reconstruction_X1[epoch].flatten(), USA.reconstruction_X2[epoch].flatten()], ['X1', 'X2'],
                  title="reconstructed data comparison")

    # plot individual loss
    plt.figure()
    plt.plot(np.array(USA.latent_variables.var_x1v1) + 0*np.random.randn(USA._n_epochs+1)/10) # add some jitter
    plt.plot(np.array(USA.latent_variables.var_x1vs)+ 0*np.random.randn(USA._n_epochs+1)/10)# add some jitter
    plt.plot(np.array(USA.latent_variables.var_x1v2)+ 0*np.random.randn(USA._n_epochs+1)/10)# add some jitter
    plt.plot(np.array(USA.latent_variables.var_x2v1)+ 0*np.random.randn(USA._n_epochs+1)/10)# add some jitter
    plt.plot(np.array(USA.latent_variables.var_x2vs)+ 0*np.random.randn(USA._n_epochs+1)/10)# add some jitter
    plt.plot(np.array(USA.latent_variables.var_x2v2)+ 0*np.random.randn(USA._n_epochs+1)/10)# add some jitter
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

    # plot individual variance across different component
    dtype = [('value', np.ndarray), ('first variance', float)]
    var_x1v1_ind=np.array([np.var(x,axis=0) for x in USA.latent_variables.x1v1])
    var_x1v1_ind=sort_helper(var_x1v1_ind,0)

    var_x1vs_ind=np.array([np.var(x,axis=0) for x in USA.latent_variables.x1vs])
    var_x1vs_ind=sort_helper(var_x1vs_ind,0)

    var_x1v2_ind=np.array([np.var(x,axis=0) for x in USA.latent_variables.x1v2])
    var_x1v2_ind=sort_helper(var_x1v2_ind,0)

    var_x2v1_ind=np.array([np.var(x,axis=0) for x in USA.latent_variables.x2v1])
    var_x2v1_ind=sort_helper(var_x2v1_ind,0)

    var_x2vs_ind=np.array([np.var(x,axis=0) for x in USA.latent_variables.x2vs])
    var_x2vs_ind=sort_helper(var_x2vs_ind,0)

    var_x2v2_ind=np.array([np.var(x,axis=0) for x in USA.latent_variables.x2v2])
    var_x2v2_ind=sort_helper(var_x2v2_ind,0)

    fig_ind,ax=plt.subplots(2,3,figsize=(10,6))
    # create colormap
    cm = plt.cm.magma(np.linspace(0, 1, hidden_size))
    ax[0][0].set_prop_cycle('color', list(cm))
    ax[0][0].plot(var_x1v1_ind)
    ax[0][0].set_title('x1v1')

    ax[0][1].set_prop_cycle('color', list(cm))
    ax[0][1].plot(var_x1vs_ind)
    ax[0][1].set_title('x1vs')

    ax[0][2].set_prop_cycle('color', list(cm))
    ax[0][2].plot(var_x1v2_ind)
    ax[0][2].set_title('x1v2')

    ax[1][0].set_prop_cycle('color', list(cm))
    ax[1][0].plot(var_x2v1_ind)
    ax[1][0].set_title('x2v1')

    ax[1][1].set_prop_cycle('color', list(cm))
    ax[1][1].plot(var_x2vs_ind)
    ax[1][1].set_title('x2vs')

    ax[1][2].set_prop_cycle('color', list(cm))
    ax[1][2].plot(var_x2v2_ind)
    ax[1][2].legend([str(i+1) for i in range(USA.hidden_size)],loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1][2].set_title('x2v2')
    plt.tight_layout()
    plt.show()


    # plot the barplot of individual variance at the last epoch
    var_x1v1_ind = np.array([np.var(x, axis=0) for x in USA.latent_variables.x1v1])
    var_x1v1_ind = sort_helper(var_x1v1_ind, -1)

    var_x1vs_ind = np.array([np.var(x, axis=0) for x in USA.latent_variables.x1vs])
    var_x1vs_ind = sort_helper(var_x1vs_ind, -1)

    var_x1v2_ind = np.array([np.var(x, axis=0) for x in USA.latent_variables.x1v2])
    var_x1v2_ind = sort_helper(var_x1v2_ind, -1)

    var_x2v1_ind = np.array([np.var(x, axis=0) for x in USA.latent_variables.x2v1])
    var_x2v1_ind = sort_helper(var_x2v1_ind, -1)

    var_x2vs_ind = np.array([np.var(x, axis=0) for x in USA.latent_variables.x2vs])
    var_x2vs_ind = sort_helper(var_x2vs_ind, -1)

    var_x2v2_ind = np.array([np.var(x, axis=0) for x in USA.latent_variables.x2v2])
    var_x2v2_ind = sort_helper(var_x2v2_ind, -1)

    all_var_bars(var_x1v1_ind,var_x1vs_ind,var_x1v2_ind,var_x2v1_ind,var_x2vs_ind,var_x2v2_ind)



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

    #all2V1=all2V1.reshape(all_shape[0],all_shape[2],hidden_size)
    #all2Vs=all2Vs.reshape(all_shape[0],all_shape[2],hidden_size)
    #all2V2=all2V2.reshape(all_shape[0],all_shape[2],hidden_size)

    print(f"var(x1v1): variance of unique component of movement preparation: {USA.latent_variables.var_x1v1[-1]:.2f})")
    print(f"var(x1vs): variance of shared component of movement preparation: {USA.latent_variables.var_x1vs[-1]:.2f})")
    print(f"var(x2v2): variance of unique component of movement execution: {USA.latent_variables.var_x2v2[-1]:.2f})")
    print(f"var(x2vs): variance of shared component of movement execution: {USA.latent_variables.var_x2vs[-1]:.2f})")

    # plot individual component for x1v1, x2v2, x1vs, x2vs (a lot of plots)
    fig, ax = plt.subplots(10, 4, figsize=(10, 12))
    max_val = np.max([x1v1, x1vs, x2vs, x2v2])
    min_val = np.min([x1v1, x1vs, x2vs, x2v2])
    fig.text(0.5, 0.003, 'time (*10 ms)', ha='center')
    fig.text(0.01, 0.5, 'amplitude (a.u.)', ha='center',rotation='vertical')
    for i in range(hidden_size):
        ax[i][0].set_prop_cycle('color', list(cm))
        ax[i][0].plot(x1v1[:, :, i].T)
        ax[i][0].set_ylim([min_val, max_val])

        ax[i][1].set_prop_cycle('color', list(cm))
        ax[i][1].plot(x1vs[:, :, i].T)
        ax[i][1].set_ylim([min_val, max_val])
        ax[i][1].set_yticks([])

        ax[i][2].set_prop_cycle('color', list(cm))
        ax[i][2].plot(x2vs[:, :, i].T)
        ax[i][2].set_ylim([min_val, max_val])
        ax[i][2].set_yticks([])

        ax[i][3].set_prop_cycle('color', list(cm))
        ax[i][3].plot(x2v2[:, :, i].T)
        ax[i][3].set_ylim([min_val, max_val])
        ax[i][3].set_yticks([])

        if i != hidden_size-1:
            ax[i][0].set_xticks([])
            ax[i][1].set_xticks([])
            ax[i][2].set_xticks([])
            ax[i][3].set_xticks([])
        if i==0:
            ax[i][0].set_title('x1v1')
            ax[i][1].set_title('x1vs')
            ax[i][2].set_title('x2vs')
            ax[i][3].set_title('x2v2')
    fig.tight_layout()
    plt.show()

    # compare top 5 component for x1vs and x2vs

    fig, ax = plt.subplots(10, 9, figsize=(10, 10))
    for h in range(hidden_size):
        for d in range(data_shape[0]):
            ax[h][d].plot(x1vs[d, :, h])
            ax[h][d].plot(x2vs[d, :, h])
            if h != hidden_size - 1:
                ax[h][d].set_xticks([])
            if h == hidden_size - 1:
                ax[h][d].set_xlabel(f'direction {d + 1}')
            if d != 0:
                ax[h][d].set_yticks([])
            if d == 0:
                ax[h][d].set_ylabel(f'corr: {np.corrcoef(x1vs[:,:,h].reshape(-1),x2vs[:,:,h].reshape(-1))[0,1]:.2f}')


    plt.legend(['x1vs', 'x2vs'], loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    plt.show()

    #
    # # jPCA analysis for latent variables
    # x1v1_all = [x for x in x1v1]
    # x1vs_all =[x for x in x1vs]
    # x2v2_all = [x for x in x2v2]
    # x2vs_all =[x for x in x2vs]
    # jpca = jPCA.JPCA(num_jpcs=2)
    # (x1v1_projected,
    #  x1v1_cps,
    #  x1v1_full_data_var,
    #  x1v1_pca_var_capt,
    #  x1v1_jpca_var_capt) = jpca.fit(x1v1_all)
    # (x1vs_projected,
    #  x1vs_cps,
    #  x1vs_full_data_var,
    #  x1vs_pca_var_capt,
    #  x1vs_jpca_var_capt) = jpca.fit(x1vs_all)
    # (x2vs_projected,
    #  x2vs_cps,
    #  x2vs_full_data_var,
    #  x2vs_pca_var_capt,
    #  x2vs_jpca_var_capt) = jpca.fit(x2vs_all)
    # (x2v2_projected,
    #  x2v2_cps,
    #  x2v2_full_data_var,
    #  x2v2_pca_var_capt,
    #  x2v2_jpca_var_capt) = jpca.fit(x2v2_all)
    #
    #
    # fig,ax=plt.subplots(2,2,figsize=(8,8))
    #
    # plot_projections(x1v1_projected,axis=ax[0][0],x_idx=0, y_idx=1,arrow_size=a_size,circle_size=c_size)
    # ax[0][0].set_title('x1v1')
    # ax[0][0].set_xlabel('jPC 1')
    # ax[0][0].set_ylabel('jPC 2')
    #
    # plot_projections(x1vs_projected,axis=ax[0][1],x_idx=0, y_idx=1,arrow_size=a_size,circle_size=c_size)
    # ax[0][1].set_title('x1vs')
    # ax[0][1].set_xlabel('jPC 1')
    # ax[0][1].set_ylabel('jPC 2')
    #
    # plot_projections(x2v2_projected,axis=ax[1][0],x_idx=0, y_idx=1,arrow_size=a_size,circle_size=c_size)
    # ax[1][0].set_title('x2v2')
    # ax[1][0].set_xlabel('jPC 1')
    # ax[1][0].set_ylabel('jPC 2')
    #
    # plot_projections(x2vs_projected,axis=ax[1][1],x_idx=0, y_idx=1,arrow_size=a_size,circle_size=c_size)
    # ax[1][1].set_title('x2vs')
    # ax[1][1].set_xlabel('jPC 1')
    # ax[1][1].set_ylabel('jPC 2')
    # fig.suptitle('latent variables jPCA analysis')
    #
    # plt.tight_layout()
    # plt.show()

    # # jPCA analysis for reconstruction
    # x1v1v1_all = [x for x in x1v1v1]
    # x1vsvs_all = [x for x in x1vsvs]
    # x2v2v2_all = [x for x in x2v2v2]
    # x2vsvs_all = [x for x in x2vsvs]
    # jpca = jPCA.JPCA(num_jpcs=2)
    # (x1v1v1_projected,
    #  _,_,_,_) = jpca.fit(x1v1v1_all)
    # (x1vsvs_projected,
    #  _,_,_,_) = jpca.fit(x1vsvs_all)
    # (x2vsvs_projected,
    #  _,_,_,_) = jpca.fit(x2vsvs_all)
    # (x2v2v2_projected,
    #  _,_,_,_) = jpca.fit(x2v2v2_all)
    #
    # fig2, ax = plt.subplots(2, 2,figsize=(8,8))
    #
    # plot_projections(x1v1v1_projected, axis=ax[0][0], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    # ax[0][0].set_title('x1v1v1')
    # ax[0][0].set_xlabel('jPC 1')
    # ax[0][0].set_ylabel('jPC 2')
    #
    # plot_projections(x1vsvs_projected, axis=ax[0][1], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    # ax[0][1].set_title('x1vsvs')
    # ax[0][1].set_xlabel('jPC 1')
    # ax[0][1].set_ylabel('jPC 2')
    #
    # plot_projections(x2v2v2_projected, axis=ax[1][0], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    # ax[1][0].set_title('x2v2v2')
    # ax[1][0].set_xlabel('jPC 1')
    # ax[1][0].set_ylabel('jPC 2')
    #
    # plot_projections(x2vsvs_projected, axis=ax[1][1], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    # ax[1][1].set_title('x2vsvs')
    # ax[1][1].set_xlabel('jPC 1')
    # ax[1][1].set_ylabel('jPC 2')
    #
    # fig2.suptitle('reconstruction jPCA analysis')
    # plt.tight_layout()
    # plt.show()

    # jPCA analysis for reconstruction
    # all2v1_all = [x for x in all2V1]
    # all2vs_all = [x for x in all2Vs]
    # all2v2_all = [x for x in all2V2]
    # jpca = jPCA.JPCA(num_jpcs=2)
    # (all2v1_projected,
    #  _,_, _, _) = jpca.fit(all2v1_all)
    # (all2vs_projected,
    #  _,_, _, _) = jpca.fit(all2vs_all)
    # (all2v2_projected,
    #  _,_, _, _) = jpca.fit(all2v2_all)
    #
    # fig3, ax = plt.subplots(2, 2,figsize=(8,8))
    #
    # plot_projections(all2v1_projected, axis=ax[0][0], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    # ax[0][0].set_title('all to V1')
    # ax[0][0].set_xlabel('jPC 1')
    # ax[0][0].set_ylabel('jPC 2')
    #
    # plot_projections(all2vs_projected, axis=ax[0][1], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    # ax[0][1].set_title('all to Vs')
    # ax[0][1].set_xlabel('jPC 1')
    # ax[0][1].set_ylabel('jPC 2')
    #
    # plot_projections(all2v2_projected, axis=ax[1][0], x_idx=0, y_idx=1, arrow_size=a_size, circle_size=c_size)
    # ax[1][0].set_title('all to V2')
    # ax[1][0].set_xlabel('jPC 1')
    # ax[1][0].set_ylabel('jPC 2')
    #
    # fig3.suptitle('all data projection jPCA analysis')
    # plt.tight_layout()
    # plt.show()


