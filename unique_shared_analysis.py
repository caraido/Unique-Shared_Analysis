
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import io
import torch
from model import fit_comp
from utils_visualization import *
from sklearn.decomposition import PCA, TruncatedSVD

def load_data(path):
    data=io.loadmat(path)

    targ_array = data['targ_array']  # Array of preparatory data (num_cond x num_neur x num_time_bins)
    move_array = data['move_array']  # Array of movement data (num_cond x num_neur x num_time_bins)
    interp_array = data['interp_array']  # Array of whole time data (num_cond x num_neur x num_time_bins)
    return targ_array, move_array, interp_array

def preprocess_data(original_data):

    pass


if __name__=='__main__':
    # load the data first
    load_folder = '/Users/tianhaolei/PycharmProjects/mini_rotation_josh/'
    path=load_folder + 'monkey_n_avgs'
    targ_array,move_array,all_array=load_data(path)

    # preprocess the data
    # "Soft normalize" each neuron's activity by dividing it by (its maximum + 5 spks/sec)

    # Get maximum activity over two epochs
    m1_max_tgt = np.max(np.max(targ_array, axis=2), axis=0)
    m1_max_move = np.max(np.max(move_array, axis=2), axis=0)
    m1_max = np.max(np.concatenate((m1_max_tgt[:, None], m1_max_move[:, None]), axis=1), axis=1)

    m1_tgt_align_norm = targ_array / (m1_max[None, :, None] + .05)  # bins are 10ms so this corresponds to 5 spks/sec
    m1_move_align_norm = move_array / (m1_max[None, :, None] + .05)

    # Subtract the mean activity across conditions (what they do in the paper)
    m1_tgt_preproc = m1_tgt_align_norm - np.mean(m1_tgt_align_norm, axis=0)[None, :, :]
    m1_move_preproc = m1_move_align_norm - np.mean(m1_move_align_norm, axis=0)[None, :, :]

    # Reshape to concatenate the conditions, so its shape (num_neurons x  time*num_conditions)
    m1_tgt_concat = m1_tgt_preproc.swapaxes(0, 1).reshape(
        [m1_tgt_preproc.shape[1], m1_tgt_preproc.shape[0] * m1_tgt_preproc.shape[2]])
    m1_move_concat = m1_move_preproc.swapaxes(0, 1).reshape(
        [m1_move_preproc.shape[1], m1_move_preproc.shape[0] * m1_move_preproc.shape[2]])

    # adjust the variance of the data
    m1_move_concat=m1_move_concat/np.std(m1_move_concat)
    m1_tgt_concat=m1_tgt_concat/np.std(m1_tgt_concat)

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
    loss, hidden, reconstructed, matrices, model = fit_comp(X, R=hidden_size, lr=0.001, n_epochs=1000,want_bias=False)

    # calculated the reconstruction loss
    reconstruct_X1=lambda x: x[0][0]+x[0][1] # X1V1V1 + X1VsVs
    reconstruct_X2=lambda x: x[1][2]+x[1][1] # X2V2V2 + X2VsVs
    reconstructed_error_X1=[np.mean(np.square(reconstruct_X1(reconstructed[i]).detach().numpy()-m1_tgt_concat.T)) for i in
                            range(len(reconstructed))] # for each epoch, calculate mean(square(X1V1V1 + X1VsVs - X1))
    reconstructed_error_X2 = [np.mean(np.square(reconstruct_X2(reconstructed[i]).detach().numpy() - m1_move_concat.T)) for i in
                              range(len(reconstructed))] # for each epoch, calculate mean(square(X2V2V2 + X2VsVs - X2))


    # get the reconstruction data estimation
    X_hat = reconstructed[-1]
    X1V1V1_hat = X_hat[0][0].detach().numpy()
    X1VsVs_hat = X_hat[0][1].detach().numpy()
    X1V2V2_hat = X_hat[0][2].detach().numpy()

    X2V1V1_hat = X_hat[1][0].detach().numpy()
    X2VsVs_hat = X_hat[1][1].detach().numpy()
    X2V2V2_hat = X_hat[1][2].detach().numpy()

    # get the transition matrices estimation
    V_hat = matrices[-1]
    V1_hat = V_hat[0].detach().numpy()
    Vs_hat = V_hat[1].detach().numpy()
    V2_hat = V_hat[2].detach().numpy()

    # reconstruct the data and plotting
    X1_hat = X1V1V1_hat + X1VsVs_hat
    X2_hat = X2V2V2_hat + X2VsVs_hat
    group_scatter([m1_tgt_concat.T.flatten(),m1_move_concat.T.flatten()], [X1_hat.flatten(), X2_hat.flatten()], ['X1', 'X2'],
                  title="reconstructed data comparison")
    # illustrating latent variables
    var_x1v1 = [np.var(z[0][0].detach().numpy()) for z in hidden]
    var_x1vs = [np.var(z[0][1].detach().numpy()) for z in hidden]
    var_x1v2 = [np.var(z[0][2].detach().numpy()) for z in hidden]
    var_x2v1 = [np.var(z[1][0].detach().numpy()) for z in hidden]
    var_x2vs = [np.var(z[1][1].detach().numpy()) for z in hidden]
    var_x2v2 = [np.var(z[1][2].detach().numpy()) for z in hidden]
    # plot individual loss
    plt.figure()
    plt.plot(var_x1v1)
    plt.plot(var_x1vs)
    plt.plot(var_x1v2)
    plt.plot(var_x2v1)
    plt.plot(var_x2vs)
    plt.plot(var_x2v2)
    #plt.plot(reconstructed_error_X1)
    #plt.plot(reconstructed_error_X2)

    plt.legend(['x1v1','x1vs','x1v2','x2v1','x2vs','x2v2'])
    plt.title('individual estimated losses-- X1:preparatory, X2: movement')
    plt.ylabel('var(estimated)')
    plt.xlabel('epoch')
    plt.show()

    plt.figure()
    plt.plot(loss)
    plt.legend(['training loss'])

    plt.figure()
    plt.plot(reconstructed_error_X1)
    plt.plot(reconstructed_error_X2)
    plt.legend(['preparatory', 'movement'])
    plt.title('reconstruction MSE')
    plt.show()

    plt.show()

    # plot the data
    # hypothesis is that there is not shared component because the movement space and preparation space are orthogonal


    pass