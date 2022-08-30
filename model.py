import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
from torch.nn import Linear, Module
import torch.nn.utils.parametrize as P
import torch.nn.utils.parametrizations as PT


class Sphere(nn.Module):
    '''
    This is used as a parametrization tool to turn a matrix into unit matrix
    '''
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x / x.norm(dim=self.dim, keepdim=True)

    def right_inverse(self, x):
        return x / x.norm(dim=self.dim, keepdim=True)


def my_loss(x1v1, x1vs, x1v2, x2v1, x2vs, x2v2, V1, Vs, V2, original_X1, original_X2,bias=False):
    '''
    :param x1v1: z1. Maximize this term
    :param x1vs: zs1. Shared component of X1. Maximize this term
    :param x1v2: projection of X1 to V2 space. Minimize this term
    :param x2v1: projection of X2 to V1 space. Minimize this term
    :param x2vs: zs2. Shared component of X2. Maximize this term
    :param x2v2: z2. Maximize this term
    :param V1: The transform matrix
    :param Vs: The transform matrx.
    :param V2: The transform matrix
    :param original_X1: original X1
    :param original_X2: original X2
    :return:
    '''
    V1_loss = torch.var(x1v1, unbiased=bias) - torch.var(x2v1, unbiased=bias)
    Vs_loss = torch.var(x1vs, unbiased=bias) + torch.var(x2vs, unbiased=bias)
    V2_loss = torch.var(x2v2, unbiased=bias) - torch.var(x1v2, unbiased=bias)

    #X1_MSE = torch.mean(torch.square(x1v1 @ V1.T + x1vs @ Vs.T - original_X1))
    #X2_MSE = torch.mean(torch.square(x2v2 @ V2.T + x2vs @ Vs.T - original_X2))

    loss = -(V1_loss + V2_loss + 0.5 * Vs_loss)# + (X1_MSE + X2_MSE)
    return loss


class CompModel(Module):
    # the process here is to create three linear layers V1, V2, Vs with a hidden size
    # the weights of V1, V2, Vs are the matrices we are looking for. There is no bias term for the linear layers
    # the input X is passed to forward() in the shape of (type x trial x time)
    # type 1 input X1 (trial x time) will be passed through the layer V1 resulting low D result x1v1
    # similarly we have x1v2, x1vs, x2v1, x2vs, x2v2
    # these latent results are used for further analysis
    # forward() method will generate x1v1 * V1.T, x1v2 * V2.T, x1vs * Vs.T, x2v1 * V1.T, x2vs * Vs.T, x2v2 * V2.T
    # Here V1.T is realized by taking the transpose of weight from linear layers V1. So are V2.T and Vs.T
    def __init__(self, input_size, hidden_size,want_bias=False):
        """
        Function that declares the model. Always in float64(double) precision
        Parameters
        ----------
        input_size: time length of a trial
            scalar
        hidden_size: number of dimensions in low-D representation
            scalar

        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.want_bias=want_bias
        # here we create a V matrix to represent [v1,vs,v2]
        assert self.input_size > self.hidden_size
        self.V = P.register_parametrization(
            PT.orthogonal(Linear(self.input_size, self.hidden_size * 3, bias=self.want_bias, dtype=torch.double)), "weight",
            Sphere(dim=-1))

    def forward(self, x):
        """
        Function that makes predictions in the model
        Parameters
        ----------
        x: input data
            3d torch tensor of shape (type x neuron x time)
        Returns
        hidden: all the latent results X * V
        output: all the terms that are needed for calculating loss X * V * V.T
        matrices: V1 Vs V2
        -------

        """
        x1 = x[0]
        x2 = x[1]

        x1v = self.V(x1)
        x2v = self.V(x2)

        x1v1 = x1v[:, 0:self.hidden_size]
        x1vs = x1v[:, self.hidden_size:2 * self.hidden_size]
        x1v2 = x1v[:, 2 * self.hidden_size:3 * self.hidden_size]

        x2v1 = x2v[:, 0:self.hidden_size]
        x2vs = x2v[:, self.hidden_size:2 * self.hidden_size]
        x2v2 = x2v[:, 2 * self.hidden_size:3 * self.hidden_size]

        V_weight = self.V.weight.T
        V1_weight = V_weight[:, 0:self.hidden_size]
        Vs_weight = V_weight[:, self.hidden_size:2 * self.hidden_size]
        V2_weight = V_weight[:, 2 * self.hidden_size:3 * self.hidden_size]

        x1v1v1 = x1v1 @ torch.transpose(V1_weight, 0, 1)
        x1vsvs = x1vs @ torch.transpose(Vs_weight, 0, 1)
        x1v2v2 = x1v2 @ torch.transpose(V2_weight, 0, 1)

        x2v1v1 = x2v1 @ torch.transpose(V1_weight, 0, 1)
        x2vsvs = x2vs @ torch.transpose(Vs_weight, 0, 1)
        x2v2v2 = x2v2 @ torch.transpose(V2_weight, 0, 1)
        hidden = ((x1v1, x1vs, x1v2), (x2v1, x2vs, x2v2))
        output = ((x1v1v1, x1vsvs, x1v2v2), (x2v1v1, x2vsvs, x2v2v2))
        matrices = (V1_weight, Vs_weight, V2_weight)
        return hidden, output, matrices


def fit_comp(X: np.ndarray, R=None, lr=0.001, n_epochs=1000,want_bias=False):
    """
        Wrapper function for fitting the neural signal comparison model
        Parameters
        ----------
        X: the input neural data (required)

        R: dimensionality of the latent (required)
           scalar
        lr: learning rate (optional)
           scalar
           Will default to 0.001
        n_epochs: number of training epochs (optional)
           scalar
           Will default to 3000
        want_bias: want bias in the model? Most likely no
        Returns
        -------
        losses: total loss for each epoch
        hiddens: X@V for each epoch
        matrices all:
            V1: matrix that help to identify the unique component of noun
            V2: matrix that help to identify the unique component of verb
            Vs: matrix that help to identify the shared component of noun & verb
        model: the pytorch model that was fit
        """
    assert R is not None
    # Include input scheduler params. currently not used
    scheduler_params = {'use_scheduler': True, 'factor': .5, 'min_lr': 5e-4, 'patience': 100, 'threshold': 1e-6,
                        'threshold_mode': 'rel'}
    x_shape = X.shape
    X = torch.tensor(X, dtype=torch.double)
    model = CompModel(input_size=x_shape[2], hidden_size=R,want_bias=want_bias)  # do we want to
    model.eval()  # "we don't want the weight the change"
    pre_train_hidden, pre_train_result, pre_train_matrices = model(
        X)  # we may need to use pre_train_hidden for analysis
    pre_train_loss = my_loss(
        pre_train_hidden[0][0],
        pre_train_hidden[0][1],
        pre_train_hidden[0][2],
        pre_train_hidden[1][0],
        pre_train_hidden[1][1],
        pre_train_hidden[1][2],
        pre_train_matrices[0],
        pre_train_matrices[1],
        pre_train_matrices[2],
        X[0], X[1],bias=want_bias
    )
    print(pre_train_loss)

    model.train()

    # set up an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # stochastic gradient descent

    # start training with the same data
    losses = np.zeros(n_epochs + 1)  # Save loss at each training epoch
    losses[0] = pre_train_loss
    hiddens = []
    hiddens.append(pre_train_hidden)
    matrices_all = []
    matrices_all.append(pre_train_matrices)
    results = []
    for e in trange(n_epochs):
        optimizer.zero_grad()
        hidden, result, matrices = model(X)
        loss = my_loss(hidden[0][0],
                       hidden[0][1],
                       hidden[0][2],
                       hidden[1][0],
                       hidden[1][1],
                       hidden[1][2],
                       matrices[0],
                       matrices[1],
                       matrices[2],
                       X[0], X[1],bias=want_bias
                       )

        loss.backward()
        optimizer.step()
        losses[e + 1] = loss
        hiddens.append(hidden)
        matrices_all.append(matrices)
        results.append(result)

    return losses, hiddens, results, matrices_all, model


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.stats import ortho_group
    from utils_visualization import group_boxplot, group_barplot, group_scatter
    from torch import from_numpy as n2t

    ########### first let's make some fake data

    record_duration = 5  # second
    sampling_rate = 100  # Hz
    num_neuron = 60  # for each trial type (what if the number of trial is different for each type?)
    R = 1  # hidden dimension

    # TODO: generate the data in the newly discussed method.
    # generate latent z
    z1 = np.random.normal(loc=0, scale=10e-1, size=(sampling_rate * record_duration, R))
    z2 = np.random.normal(loc=0, scale=10e-1, size=(sampling_rate * record_duration, R))
    # standard normal distribition
    # zs1 = np.random.randn(sampling_rate * record_duration,R)
    # zs2 = np.random.randn(sampling_rate * record_duration, R)
    # all zeros
    # zs1=np.zeros([sampling_rate * record_duration,R])
    # zs2 = np.zeros([sampling_rate * record_duration, R])
    # normal distribution centered at 0 with very smaller var
    zs1 = np.random.normal(loc=0, scale=10e-2, size=(sampling_rate * record_duration, R))
    zs2 = np.random.normal(loc=0, scale=10e-2, size=(sampling_rate * record_duration, R))
    z_rand = np.random.randn(sampling_rate * record_duration, R)

    # generate random orthogonal V and get V1, V2, V3 from it
    V = ortho_group.rvs(dim=num_neuron)
    V1 = V[:, 0:R]
    V2 = V[:, R:2 * R]
    Vs = V[:, 2 * R:3 * R]
    V_rand = V[:, 4 * R:5 * R]

    V1_orth = V1.T @ V1
    V2_orth = V2.T @ V2
    Vs_orth = Vs.T @ Vs
    print(f"check orthogonality of V1: V1.T@V1={V1_orth}")
    print(f"check orthogonality of Vs: Vs.T@Vs={Vs_orth}")
    print(f"check orthogonality of V2: V2.T@V2={V2_orth}")

    # generate X1 X2
    X1 = z1 @ V1.T + zs1 @ Vs.T
    X2 = z2 @ V2.T + zs2 @ Vs.T

    # center the data
    X1 = X1
    X2 = X2

    # make X
    X = np.array([X1, X2])

    # train the model and calculate the true loss
    loss, hidden, reconstructed, matrices, model = fit_comp(X, R, lr=0.001, n_epochs=800)
    true_loss = my_loss(n2t(X1 @ V1), n2t(X1 @ Vs), n2t(X1 @ V2), n2t(X2 @ V1), n2t(X2 @ Vs), n2t(X2 @ V2), V1, Vs,
                        V2,X1,X2).numpy()

    # get the latent variable estimation
    z_hat = hidden[-1]
    z1_hat = z_hat[0][0].detach().numpy()
    zs1_hat = z_hat[0][1].detach().numpy()
    z2_hat = z_hat[1][2].detach().numpy()
    zs2_hat = z_hat[1][1].detach().numpy()

    # get the transition matrices estimation
    V_hat = matrices[-1]
    V1_hat = V_hat[0].detach().numpy()
    Vs_hat = V_hat[1].detach().numpy()
    V2_hat = V_hat[2].detach().numpy()

    # get the reconstruction data estimation
    X_hat = reconstructed[-1]
    X1V1V1_hat = X_hat[0][0].detach().numpy()
    X1VsVs_hat = X_hat[0][1].detach().numpy()
    X1V2V2_hat = X_hat[0][2].detach().numpy()

    X2V1V1_hat = X_hat[1][0].detach().numpy()
    X2VsVs_hat = X_hat[1][1].detach().numpy()
    X2V2V2_hat = X_hat[1][2].detach().numpy()

    # reconstruct the data
    X1_hat = X1V1V1_hat + X1VsVs_hat
    X2_hat = X2V2V2_hat + X2VsVs_hat

    # (x1v1v1, x1vsvs, x1v2v2), (x2v1v1, x2vsvs, x2v2v2)
    ########### post processing #############

    # align the latent variables and transition matrix
    z1_hat = -z1_hat if np.corrcoef(z1_hat.flatten(), z1.flatten())[0, -1] < 0 else z1_hat
    zs1_hat = -zs1_hat if np.corrcoef(zs1_hat.flatten(), zs1.flatten())[0, -1] < 0 else zs1_hat
    zs2_hat = -zs2_hat if np.corrcoef(zs2_hat.flatten(), zs2.flatten())[0, -1] < 0 else zs2_hat
    z2_hat = -z2_hat if np.corrcoef(z2_hat.flatten(), z2.flatten())[0, -1] < 0 else z2_hat

    V1_hat = -V1_hat if np.corrcoef(V1_hat.flatten(), V1.flatten())[0, -1] < 0 else V1_hat
    Vs_hat = -Vs_hat if np.corrcoef(Vs_hat.flatten(), Vs.flatten())[0, -1] < 0 else Vs_hat
    V2_hat = -V2_hat if np.corrcoef(V2_hat.flatten(), V2.flatten())[0, -1] < 0 else V2_hat

    # calculate the matrix norm of original vs estimate
    z1_diff = np.linalg.norm(z1 - z1_hat) / (sampling_rate * record_duration * R)
    z2_diff = np.linalg.norm(z2 - z2_hat) / (sampling_rate * record_duration * R)
    zs1_diff = np.linalg.norm(zs1 - zs1_hat) / (sampling_rate * record_duration * R)
    zs2_diff = np.linalg.norm(zs2 - zs2_hat) / (sampling_rate * record_duration * R)

    V1_diff = np.linalg.norm(V1_hat - V1) / (R * num_neuron)
    V2_diff = np.linalg.norm(V2_hat - V2) / (R * num_neuron)
    Vs_diff = np.linalg.norm(Vs_hat - Vs) / (R * num_neuron)

    ##### compare to random #######
    z1_rand_diff = np.linalg.norm(z_rand - z1_hat) / (sampling_rate * record_duration * R)
    z2_rand_diff = np.linalg.norm(z_rand - z2_hat) / (sampling_rate * record_duration * R)
    zs1_rand_diff = np.linalg.norm(z_rand - zs1_hat) / (sampling_rate * record_duration * R)
    zs2_rand_diff = np.linalg.norm(z_rand - zs2_hat) / (sampling_rate * record_duration * R)

    V1_rand_diff = np.linalg.norm(V1_hat - V_rand) / (R * num_neuron)
    V2_rand_diff = np.linalg.norm(V2_hat - V_rand) / (R * num_neuron)
    Vs_rand_diff = np.linalg.norm(Vs_hat - V_rand) / (R * num_neuron)

    # compare similarity between zs, zs_1_hat and zs_2_hat
    print(
        f"norm between zs_1_hat and zs_2_hat: {np.linalg.norm(zs1_hat - zs2_hat):.3f}")
    print(
        f"norm between z1(x1v1) and z1_hat: {np.linalg.norm(z1 - z1_hat):.3f}")
    print(
        f"norm between z2(x2v2) and z2_hat: {np.linalg.norm(z2 - z2_hat) :.3f}")
    print(
        f"norm between zs1(x1vs) and zs1_hat: {np.linalg.norm(zs1_hat - zs1):.3f}")
    print(
        f"norm between zs2(x2vs) and zs2_hat: {np.linalg.norm(zs2_hat - zs2):.3f}")

    print(f"norm between V1 and V1_hat: {np.linalg.norm(V1_hat - V1):.3f}")
    print(f"norm between V2 and V2_hat: {np.linalg.norm(V2_hat - V2):.3f}")
    print(f"norm between Vs and Vs_hat: {np.linalg.norm(Vs_hat - Vs):.3f}")

    print(f"variance of X1:{np.var(X1)}")
    print(f"variance of X2:{np.var(X2)}")

    print(f'mean of X1: {np.mean(X1)}')
    print(f'mean of X2: {np.mean(X2)}')

    print(f"true loss: {true_loss}")
    print(f"estimated loss- true loss: {loss[-1] - true_loss}")
    # plotting

    ### this is the boxplot for the latent variables
    z_original = np.array([z1, zs1, zs2, z2]).reshape(4, -1)
    z_estimate = np.array([z1_hat, zs1_hat, zs2_hat, z2_hat]).reshape(4, -1)
    z_labels = ['z1', 'zs1', 'zs2', 'z2']
    group_boxplot(z_original, z_estimate, labels_list=z_labels, title='latent variables estimate comparison',
                  legend=['original', 'estimate'])

    ### this is the boxplot for transition matrix
    V_original = np.array([V1, Vs, V2]).reshape(3, -1)
    V_estimate = np.array([V1_hat, Vs_hat, V2_hat]).reshape(3, -1)
    V_labels = ['V1', 'Vs', 'V2']
    # group_boxplot(V_original, V_estimate, labels_list=V_labels, title='transition matrices estimate comparison',
    #              legend=['original', 'estimate'])

    ### this is the barplot for the latent variables
    z_diff = np.array([z1_diff, zs1_diff, zs2_diff, z2_diff])
    z_rand_diff = np.array([z1_rand_diff, zs1_rand_diff, zs2_rand_diff, z2_rand_diff])
    # group_barplot(z_diff,z_rand_diff,labels_list=z_labels,title='latent variables estimate comparison',legend=['estimate vs original', 'estimate vs random'])

    ### this is the barplot for transition matrix
    V_diff = np.array([V1_diff, Vs_diff, V2_diff])
    V_rand_diff = np.array([V1_rand_diff, Vs_rand_diff, V2_rand_diff])
    # group_barplot(V_diff, V_rand_diff, labels_list=V_labels, title='transition matrices estimate comparison',
    #              legend=['estimate vs original', 'estimate vs random'])

    ### these are the scatter plots to compare z, V and reconstruction
    group_scatter(V_original, V_estimate, V_labels, title='transition matrices estimate comparison')
    group_scatter(z_original, z_estimate, z_labels, title='latent variables estimate comparison')
    group_scatter([X1.flatten(), X2.flatten()], [X1_hat.flatten(), X2_hat.flatten()], ['X1', 'X2'],
                  title="reconstructed data comparison")

    ### plot individual loss difference (estimated loss vs ground truth loss)
    # here we have var(x1v1), var(x1vs), var(x1v2),var(x2v1),var(x2vs), var(x2v2) vs there ground truth
    x1v1_true = z1
    x1vs_true = X1 @ Vs
    x1v2_true = X1 @ V2
    x2v1_true = X2 @ V1
    x2vs_true = X2 @ Vs
    x2v2_true = z2

    var_x1v1 = [-np.var(z[0][0].detach().numpy()) + np.var(x1v1_true) for z in hidden]
    var_x1vs = [-np.var(z[0][1].detach().numpy()) + np.var(x1vs_true) for z in hidden]
    var_x1v2 = [-np.var(z[0][2].detach().numpy()) + np.var(x1v2_true) for z in hidden]
    var_x2v1 = [-np.var(z[1][0].detach().numpy()) + np.var(x2v1_true) for z in hidden]
    var_x2vs = [-np.var(z[1][1].detach().numpy()) + np.var(x2vs_true) for z in hidden]
    var_x2v2 = [-np.var(z[1][2].detach().numpy()) + np.var(x2v2_true) for z in hidden]

    var_x2vsvs = [np.var(z[1][2].detach().numpy() @ v[1].detach().numpy().T) for z, v in zip(hidden, matrices)]
    var_x1vsvs = [np.var(z[0][1].detach().numpy() @ v[1].detach().numpy().T) for z, v in
                  zip(hidden, matrices)]
    var_x1v1v1 = [np.var(z[0][0].detach().numpy() @ v[0].detach().numpy().T) for z, v in zip(hidden, matrices)]
    var_x2v2v2 = [np.var(z[1][2].detach().numpy() @ v[2].detach().numpy().T) for z, v in
                  zip(hidden, matrices)]

    # plot individual loss
    plt.figure()
    plt.plot(var_x1v1)
    plt.plot(var_x1vs)
    plt.plot(var_x1v2)
    plt.plot(var_x2v1)
    plt.plot(var_x2vs)
    plt.plot(var_x2v2)

    plt.legend(['x1v1', 'x1vs', 'x1v2', 'x2v1', 'x2vs', 'x2v2'])
    plt.title('individual losses: true loss-estimated loss')
    plt.ylabel('var(true)-var(estimated)')
    plt.xlabel('epoch')
    plt.show()

    # plot individual loss for reconstruction
    plt.plot(var_x1v1v1)
    plt.plot(var_x1vsvs)

    plt.plot(var_x2v2v2)
    plt.plot(var_x2vsvs)
    plt.legend(['z1v1', 'zs1vs', 'z2v2', 'zs2vs'])
    plt.title('individual reconstruction losses: true loss-estimated loss')
    plt.ylabel('var(true)-var(estimated)')
    plt.xlabel('epoch')
    plt.show()

    plt.figure()
    plt.plot(loss)
    plt.axhline(y=true_loss, color='red')

    plt.legend(['training loss', 'true loss'])
    plt.show()