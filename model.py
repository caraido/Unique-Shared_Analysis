import numpy as np
from tqdm import trange
import warnings
import torch
import torch.nn as nn
from torch.nn import Linear, Module
import torch.nn.utils.parametrize as P
import torch.nn.utils.parametrizations as PT
from sklearn.decomposition import PCA, TruncatedSVD

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
    Vs_loss = torch.std(x1vs, unbiased=bias) * torch.std(x2vs, unbiased=bias)
    V2_loss = torch.var(x2v2, unbiased=bias) - torch.var(x1v2, unbiased=bias)

    #X1_MSE = torch.mean(torch.square(x1v1 @ V1.T + x1vs @ Vs.T - original_X1))
    #X2_MSE = torch.mean(torch.square(x2v2 @ V2.T + x2vs @ Vs.T - original_X2))

    loss = -(V1_loss + V2_loss + 1 * Vs_loss)# + (X1_MSE + X2_MSE)
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
    def __init__(self, input_size, hidden_size,want_bias=False,V_init=None):
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
        self.V_init=V_init
        # here we create a V matrix to represent [v1,vs,v2]
        assert self.input_size >= self.hidden_size

        if self.V_init is not None:
            assert V_init[0].shape == (self.input_size, self.hidden_size)
            self.V_init=np.concatenate(self.V_init,axis=1)
            VV=Linear(self.input_size, self.hidden_size * 3, bias=self.want_bias, dtype=torch.double)
            VV.weight=nn.Parameter(torch.DoubleTensor(self.V_init).T)
            self.V=P.register_parametrization(PT.orthogonal(VV), "weight",Sphere(dim=-1))

            original_weight=self.V_init.T # inspect this to make sure
            after_weight=self.V.weight.detach().numpy()
        else:
            VV=Linear(self.input_size, self.hidden_size * 3, bias=self.want_bias, dtype=torch.double)
            original_weight = VV.weight.detach().numpy()
            self.V = P.register_parametrization(
                PT.orthogonal(VV),
                "weight",
                Sphere(dim=-1))

            after_weight=self.V.weight.detach().numpy()


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


class LatentVariables:
    def __init__(self,hidden):
        # self.x1v1 = [np.float(z[0][0].detach().numpy()) for z in hidden]
        # self.x1vs = [np.float(z[0][1].detach().numpy()) for z in hidden]
        # self.x1v2 = [np.float(z[0][2].detach().numpy()) for z in hidden]
        # self.x2v1 = [np.float(z[1][0].detach().numpy()) for z in hidden]
        # self.x2vs = [np.float(z[1][1].detach().numpy()) for z in hidden]
        # self.x2v2 = [np.float(z[1][2].detach().numpy()) for z in hidden]
        self.x1v1 = [z[0][0] for z in hidden]
        self.x1vs = [z[0][1] for z in hidden]
        self.x1v2 = [z[0][2] for z in hidden]
        self.x2v1 = [z[1][0] for z in hidden]
        self.x2vs = [z[1][1] for z in hidden]
        self.x2v2 = [z[1][2] for z in hidden]

        self.var_x1v1=[np.var(h) for h in self.x1v1]
        self.var_x1vs = [np.var(h) for h in self.x1vs]
        self.var_x1v2 = [np.var(h) for h in self.x1v2]
        self.var_x2v1 = [np.var(h) for h in self.x2v1]
        self.var_x2vs = [np.var(h) for h in self.x2vs]
        self.var_x2v2 = [np.var(h) for h in self.x2v2]


class Reconstructed:
    def __init__(self,reconstructed):
        #((x1v1v1, x1vsvs, x1v2v2), (x2v1v1, x2vsvs, x2v2v2))
        #self.X1V1V1=[np.float(r[0][0].detach().numpy()) for r in reconstructed]
        #self.X1VsVs = [np.float(r[0][1].detach().numpy()) for r in reconstructed]
        #self.X1V2V2 = [np.float(r[0][2].detach().numpy()) for r in reconstructed]
        #self.X2V1V1 = [np.float(r[1][0].detach().numpy()) for r in reconstructed]
        #self.X2VsVs = [np.float(r[1][1].detach().numpy()) for r in reconstructed]
        #self.X2V2V2 = [np.float(r[1][2].detach().numpy()) for r in reconstructed]
        self.X1V1V1=[r[0][0] for r in reconstructed]
        self.X1VsVs = [r[0][1] for r in reconstructed]
        self.X1V2V2 = [r[0][2] for r in reconstructed]
        self.X2V1V1 = [r[1][0] for r in reconstructed]
        self.X2VsVs = [r[1][1] for r in reconstructed]
        self.X2V2V2 = [r[1][2] for r in reconstructed]

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
        #self.V1=[np.float(v[0].detach().numpy()) for v in mat]
        #self.Vs = [np.float(v[1].detach().numpy()) for v in mat]
        #self.V2 = [np.float(v[2].detach().numpy()) for v in mat]
        self.V1=mat[0]
        self.Vs=mat[1]
        self.V2=mat[2]


class Initialization:
    def __init__(self,method=None):
        self._method=None
        self.method=method
        self.V=None
        self.tracking=[]

    @property
    def method(self):
        return self._method
    @method.setter
    def method(self,method):
        if method is not None and method !='iter':
            raise ValueError('method can only be None or iter for now')


class UniqueSharedAnalysis:

    def __init__(self,hidden_size):
        self._hidden_size=None
        self._raw_data=None
        self._raw_data_size=None
        self._lr=None
        self._n_epochs=None
        self._want_bias=None
        self._want_stats=False
        self.initialization=Initialization()

        self.run = False
        self.hidden_size=hidden_size
        self.losses=None
        self.latent_variables=None
        self.reconstructed=None
        self.transition_mat=None
        self.model=None
        self.X1=None
        self.X2=None

    def initialize(self,data,method='iter'):
        '''test this on the fake data before proceed'''
        if self.check_format(data) and method=='iter':
            self.initialization.method=method

            X1=data[0]
            X2=data[1]

            V1=[]
            Vs=[]
            V2=[]
            all_V=[V1,Vs,V2]
            svd = TruncatedSVD(n_components=self.hidden_size)

            pick=1
            tracking=[]
            for i in range(3*self.hidden_size):
                if pick==1:
                    svd.fit(X1)
                else:
                    svd.fit(X2)
                pick*=-1
                W=svd.components_[0,None].T

                V1_pref=np.var(X1@W)-np.var(X2@W)
                Vs_pref=np.std(X1@W)*np.std(X2@W)
                #Vs_pref=(np.var(X1@W) + np.var(X2@W))/2
                V2_pref=np.var(X2@W)-np.var(X1@W)

                order = np.argsort([V1_pref, Vs_pref, V2_pref])[::-1]  # descending order

                for o in order:
                    if len(all_V[o])<self.hidden_size:
                        all_V[o].append(W)
                        tracking.append(o)
                        break

                X1 = X1 - X1 @ W @ W.T
                X2 = X2 - X2 @ W @ W.T

            self.initialization.tracking=tracking
            print(tracking)

            V1=np.concatenate(V1,axis=-1)
            Vs=np.concatenate(Vs,axis=-1)
            V2=np.concatenate(V2,axis=-1)
            # check orthogonality here TODO: double check the assertions
            if 0:
                if np.sum(X1@V1)>10e-10 and np.sum(X2@V2)>10e-10:# in case the unique component doesn't exist

                    assert np.sum(V1.T@V1-np.eye(self.hidden_size))<10e-10
                    assert np.sum(V2.T @ V2 - np.eye(self.hidden_size))<10e-10
                    assert np.sum(Vs.T @ Vs - np.eye(self.hidden_size))<10e-10

                    assert np.sum(V2.T @ V1 ) < 10e-10
                    assert np.sum(Vs.T @ V1) < 10e-10
                    assert np.sum(V2.T @ Vs) < 10e-10

            self.initialization.V=[V1,Vs,V2]

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

            loss, hidden, reconstructed, matrices, model = fit_comp(self._raw_data, V_init=self.initialization.V,R=self.hidden_size, lr=self._lr, n_epochs=self._n_epochs,
                                                                    want_bias=self._want_bias)
            self.run=True

            self.losses=loss
            #del loss
            self.latent_variables=LatentVariables(hidden)
            #del hidden # to save the memory
            self.reconstructed=Reconstructed(reconstructed)
            #del reconstructed # to save the memory
            self.transition_mat=TransitionMat(matrices)
            #del matrices # to save the memory
            self.model=model

            if self._want_stats:
                self._calculate_stats()

            print(f"Original total loss:{self.losses[0]:.2f}. ")
            print('### after training ###')
            print(f"Last epoch total loss:{self.losses[-1]:.2f}. ")
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
                self._raw_data_size=size
                return True
            else: return False
        elif isinstance(data,torch.Tensor):
            size=data.size()
            if size[0]==2 and len(size)==3:
                self._raw_data_size=size
                data=data.detach().numpy()
                warnings.warn(f"Preferred numpy array data type but got {data.type()}. Now transforming to numpy array")
                return True
            else: return False
        elif isinstance(data,list):
            if len(data)==2 :
                self._raw_data_size=(data[0].shape,data[1].shape)
                return True
            else:
                return False
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


def fit_comp(X, R=None, V_init:list=None, lr=0.001, n_epochs=1000,want_bias=False):
    """
        Wrapper function for fitting the neural signal comparison model
        Parameters
        ----------
        X: the input neural data (required)
            usually include [X1, X2] with shape of X1: time1 x n_neuron, X2: time2 x n_neuron. time1 and time2 may be equal
        R: dimensionality of the latent (required)
           scalar
        V_init: initialized V, need to check orthogonality
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
    # TODO: revisit this
    if 0:
        [V1,Vs,V2]= V_init
        assert np.sum(V1@V1.T-np.eye(len(V1)))<10e-7 or np.sum(V1.T@V1-np.eye(len(V1.T)))<10e-7
        assert np.sum(Vs@Vs.T-np.eye(len(Vs)))<10e-7 or np.sum(Vs.T@Vs-np.eye(len(Vs.T)))<10e-7
        assert np.sum(V2@V1.T-np.eye(len(V2)))<10e-7 or np.sum(V2.T@V2-np.eye(len(V2.T)))<10e-7
        assert np.sum(V2 @ V1.T )<10e-7 or np.sum(V2.T@V1)<10e-7
        assert np.sum(Vs @ V1.T) < 10e-7 or np.sum(Vs.T@V1)<10e-7
        assert np.sum(V2 @ Vs.T) < 10e-7 or np.sum(V2.T@Vs)<10e-7

    # Include input scheduler params. currently not used
    scheduler_params = {'use_scheduler': True, 'factor': .5, 'min_lr': 5e-4, 'patience': 100, 'threshold': 1e-6,
                        'threshold_mode': 'rel'}

    X1_input_size=X[0].shape
    X2_input_size=X[1].shape
    assert X1_input_size[1]==X2_input_size[1]
    input_size=X1_input_size[1]

    X = [torch.tensor(X[0], dtype=torch.double),torch.tensor(X[1], dtype=torch.double)]
    model = CompModel(input_size=input_size, hidden_size=R,want_bias=want_bias,V_init=V_init)  # do we want to
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


    model.train()

    # set up an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # stochastic gradient descent

    # start training with the same data
    losses = np.zeros(n_epochs + 1)  # Save loss at each training epoch
    losses[0] = np.float(pre_train_loss)
    pre_train_hidden = [[h.detach().numpy().astype(np.float32) for h in hid] for hid in pre_train_hidden]
    pre_train_matrices = [mat.detach().numpy().astype(np.float32) for mat in pre_train_matrices]
    pre_train_result = [[r.detach().numpy().astype(np.float32) for r in rst] for rst in pre_train_result]

    hiddens = []
    hiddens.append(pre_train_hidden)
    matrices_all = []
    matrices_all.append(pre_train_matrices)
    results = []
    results.append(pre_train_result)
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

        hidden=[[h.detach().numpy().astype(np.float32) for h in hid] for hid in hidden]
        matrices = [mat.detach().numpy().astype(np.float32) for mat in matrices]
        result = [[r.detach().numpy().astype(np.float32) for r in rst] for rst in result]

        losses[e + 1] = np.float(loss)
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
    R = 2  # hidden dimension

    # TODO: generate the data in the newly discussed method.
    # generate latent z
    z1 = np.random.normal(loc=0, scale=10e-1, size=(sampling_rate * record_duration, R))
    z2 = np.random.normal(loc=0, scale=10e-1, size=(sampling_rate * record_duration, R))
    zs1 = np.random.normal(loc=0, scale=10e-1, size=(sampling_rate * record_duration, R))
    zs2 = np.random.normal(loc=0, scale=10e-1, size=(sampling_rate * record_duration, R))
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
    true_loss = my_loss(n2t(X1 @ V1), n2t(X1 @ Vs), n2t(X1 @ V2), n2t(X2 @ V1), n2t(X2 @ Vs), n2t(X2 @ V2), V1, Vs,
                        V2, X1, X2).numpy()

    # train the model and calculate the true loss
    usa=UniqueSharedAnalysis(hidden_size=R)
    usa.initialize(X)
    usa.fit(X,n_epochs=1000)
    epoch=-1

    # calculate the matrix norm of original vs estimate
    z1_diff = np.linalg.norm(z1 - usa.latent_variables.x1v1[epoch]) / (sampling_rate * record_duration * R)
    z2_diff = np.linalg.norm(z2 - usa.latent_variables.x2v2[epoch]) / (sampling_rate * record_duration * R)
    zs1_diff = np.linalg.norm(zs1 - usa.latent_variables.x1vs[epoch]) / (sampling_rate * record_duration * R)
    zs2_diff = np.linalg.norm(zs2 - usa.latent_variables.x2vs[epoch]) / (sampling_rate * record_duration * R)

    V1_diff = np.linalg.norm(usa.transition_mat.V1[epoch] - V1) / (R * num_neuron)
    V2_diff = np.linalg.norm(usa.transition_mat.V2[epoch] - V2) / (R * num_neuron)
    Vs_diff = np.linalg.norm(usa.transition_mat.Vs[epoch] - Vs) / (R * num_neuron)

    ##### compare to random #######
    z1_rand_diff = np.linalg.norm(z_rand - usa.latent_variables.x1v1[epoch]) / (sampling_rate * record_duration * R)
    z2_rand_diff = np.linalg.norm(z_rand - usa.latent_variables.x2v2[epoch]) / (sampling_rate * record_duration * R)
    zs1_rand_diff = np.linalg.norm(z_rand - usa.latent_variables.x1vs[epoch]) / (sampling_rate * record_duration * R)
    zs2_rand_diff = np.linalg.norm(z_rand - usa.latent_variables.x2vs[epoch]) / (sampling_rate * record_duration * R)

    V1_rand_diff = np.linalg.norm(usa.transition_mat.V1[epoch] - V_rand) / (R * num_neuron)
    V2_rand_diff = np.linalg.norm(usa.transition_mat.V2[epoch] - V_rand) / (R * num_neuron)
    Vs_rand_diff = np.linalg.norm(usa.transition_mat.Vs[epoch] - V_rand) / (R * num_neuron)

    # compare similarity between zs, zs_1_hat and zs_2_hat
    print(f"variance of X1:{np.var(X1)}")
    print(f"variance of X2:{np.var(X2)}")

    print(f'mean of X1: {np.mean(X1)}')
    print(f'mean of X2: {np.mean(X2)}')

    print(f"true loss: {true_loss}")
    print(f"estimated loss- true loss: {usa.losses[epoch] - true_loss}")
    # plotting

    ### this is the boxplot for the latent variables
    z_original = np.array([z1, zs1, zs2, z2]).reshape(4, -1)
    z_estimate = np.array([usa.latent_variables.x1v1[epoch], usa.latent_variables.x1vs[epoch], usa.latent_variables.x2vs[epoch], usa.latent_variables.x2v2[epoch]]).reshape(4, -1)
    z_labels = ['z1', 'zs1', 'zs2', 'z2']
    group_boxplot(z_original, z_estimate, labels_list=z_labels, title='latent variables estimate comparison',
                  legend=['original', 'estimate'])

    ### this is the boxplot for transition matrix
    V_original = np.array([V1, Vs, V2]).reshape(3, -1)
    V_estimate = np.array([usa.transition_mat.V1[epoch], usa.transition_mat.Vs[epoch], usa.transition_mat.V2[epoch]]).reshape(3, -1)
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
    group_scatter([X1.flatten(), X2.flatten()], [usa.reconstruction_X1[epoch].flatten(), usa.reconstruction_X2[epoch].flatten()], ['X1', 'X2'],
                  title="reconstructed data comparison")

    ### plot individual loss difference (estimated loss vs ground truth loss)
    # here we have var(x1v1), var(x1vs), var(x1v2),var(x2v1),var(x2vs), var(x2v2) vs there ground truth
    x1v1_true = z1
    x1vs_true = X1 @ Vs
    x1v2_true = X1 @ V2
    x2v1_true = X2 @ V1
    x2vs_true = X2 @ Vs
    x2v2_true = z2

    # plot individual loss
    plt.figure()
    plt.plot(usa.latent_variables.var_x1v1)
    plt.plot(usa.latent_variables.var_x1vs)
    plt.plot(usa.latent_variables.var_x1v2)
    plt.plot(usa.latent_variables.var_x2v1)
    plt.plot(usa.latent_variables.var_x2vs)
    plt.plot(usa.latent_variables.var_x2v2)

    plt.legend(['x1v1', 'x1vs', 'x1v2', 'x2v1', 'x2vs', 'x2v2'])
    plt.title('individual losses: true loss-estimated loss')
    plt.ylabel('var(true)-var(estimated)')
    plt.xlabel('epoch')
    plt.show()

    # plot individual loss for reconstruction
    plt.figure()
    plt.plot(usa.X1_recon_MSE)
    plt.plot(usa.X2_recon_MSE)
    plt.legend(['X1', 'X2'])
    plt.title('reconstruction MSE')
    plt.show()

    plt.figure()
    plt.plot(usa.losses)
    plt.axhline(y=true_loss, color='red')

    plt.legend(['training loss', 'true loss'])
    plt.show()
