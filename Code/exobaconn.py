import numpy as np
import tensorflow as tf
tfk = tf.keras
tfk.backend.set_floatx("float64")
import keras.backend as K
from math import ceil
from spectres import spectres
import sklearn as sk
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from corner import corner
import pickle
import astropy.constants as cte
import astropy.units as U

def read_grid(file_with_spectra, file_with_parameters, wvl=None, new_wvl=None, 
              n_train=9000, n_test=1000, N_aug=[1,1]):
    
    '''
    This function reads in two .npy files with the spectra and corresponding parameters and returns features
    and labels arrays for training and testing.
    
    Arguments: 
    ----------
    file_with_spectra
    file_with_parameters
    wvl
    new_wvl
    n_train
    n_test
    N_aug
    
    Outputs:
    --------
    X_train
    Y_train
    X_test
    Y_test
    '''
    
    print('Loading files...')
    X = np.load(file_with_spectra)
    Y = np.load(file_with_parameters)
    
    if type(new_wvl)==np.ndarray:
        print('Rebinning spectra')
        X = spectres(new_wvl, wvl, X)
    
    print('X_train...')
    X_train = np.repeat(X[:n_train, :], N_aug[0], axis=0)
    print('Y_train...')
    Y_train = np.repeat(Y[:n_train, :], N_aug[0], axis=0)
    
    print('X_test...')
    X_test = np.repeat(X[-n_test:, :], N_aug[1], axis=0)
    print('Y_test...')
    Y_test = np.repeat(Y[-n_test:, :], N_aug[1], axis=0)
    
    return X_train, Y_train, X_test, Y_test

def add_noise(spectra, err, floor=0):
    '''
    Adds the noise specified in err to spectra. The noise needs to be in ppm.
    err can be a float, in which case the same noise will be applied to the whole spectra; an array of the same length as the spectra,
    specifying the noise at each wavelength bin; or a string with the name of a file containing the wavelengths in the first column and the 
    corresponding noise in the second.
    
    A noise floor can be specified (in ppm) in case err does not include it.
    '''
    
    if type(err) == np.ndarray:
        err[err < floor] = floor
        noise = 1e-4*err.reshape(1,-1)*np.random.randn(spectra.shape[0], spectra.shape[1])
        noisy_spec = spectra + noise
    elif type(err) == float:
        err = np.max([err, floor])
        noise = 1e-4*err*np.random.randn(spectra.shape[0], spectra.shape[1])
        noisy_spec = spectra + noise
    elif type(err) == str:
        err = np.loadtxt(err)[:,1]
        err[err < floor] = floor
        noise = 1e-4*err.reshape(1,-1)*np.random.randn(spectra.shape[0], spectra.shape[1])
        noisy_spec = spectra + noise
    else:
        print('Noise not understood!')
        
    return noisy_spec
    
def normalize(X_train, method, conc):
    '''
    Methods (str):
    --------
    'meanstd' : (X - mean) / std
    '-min'    : X - min(X)
    'minmax'  : (X - min(X)) / (max(X) - min(X))
    --------------------------------------------
    
    If conc is True, then the function will concatenate the normalized spectrum to the original one.
    '''

    def mean(X_train):
        return X_train.mean(1).reshape(len(X_train), 1)
    def std(X_train):
        std = X_train.std(1).reshape(len(X_train), 1)
        return std
    
    if method == '-meanstd_0':
        X_train_n = (X_train-X_train.mean(0))/X_train.std(0)
    elif method == '-meanstd_1':
        std = std(X_train)
        X_train_n = (X_train-mean(X_train))/std
        sel = std<1e-8
        sel = sel[:,0]
        X_train_n[sel] = np.zeros(len(X_train[0]))
    elif method == '-mean':
        X_train_n = X_train-mean(X_train)
    elif method == 'standardize':
        X_train_n = StandardScaler().fit_transform(X_train)
    else:
        print('The normalization method is not valid.')
              
    if conc>=0 and conc<=X_train.ndim:
        X_train = np.expand_dims(X_train, 2)
        X_train_n = np.expand_dims(X_train_n, 2)
        X_train_n = np.concatenate((X_train, X_train_n), axis=conc)
    elif conc == -1:
        X_train_n = np.expand_dims(X_train_n, 2)

    return X_train_n

def true_vs_pred(true, pred, names, color_code=None, a=0.05, sup_title=False, dims_j=3 , ys='all'):
    D = true.shape[1]
    
    if ys=='all':
        ys=range(D)
        
    mins = true.min(0)
    maxs = true.max(0)

    R2 = [sk.metrics.r2_score(true[:,i], pred[:,i]) for i in range(D)]
    bias = [np.mean((pred[:,i]-true[:,i]), axis=0) for i in range(D)]

    dims_i = max(2, ceil(len(ys)/2))
        
    fig = plt.figure(figsize=(7*dims_j, 6*dims_i))
    
    handles = [Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 2
    
    k=0
    for i in range(len(ys)):
        p=ys[i]
        ax = fig.add_subplot(dims_i, dims_j, 1+i)
        if color_code==None:
            im=ax.scatter(true[:,p], pred[:,p], c='royalblue', alpha=a)
        else:
            im=ax.scatter(true[:,p], pred[:,p],c=true[:,color_code],cmap="RdYlGn",
                                vmin=min(true[:,color_code]), 
                                vmax=max(true[:,color_code]),s=10, alpha=a)
            cbar = fig.colorbar(im, ax=axs[i,j])
            cbar.set_label(names[color_code])
            cbar.solids.set(alpha=1)
        ax.plot(np.sort(true[:,p]),np.sort(true[:,p]),'r--')
        ax.set_xlim([mins[p],maxs[p]])
        ax.set_ylim([mins[p],maxs[p]])
        ax.set_xlabel('True '+names[p], fontsize='x-large')
        ax.set_ylabel('Predicted '+names[p], fontsize='x-large')
        ax.grid(True)
        labels=[r'$R^2 = $'+str(round(R2[p],2)), r'$MB = $'+str(round(bias[p],2))]
        ax.legend(handles, labels, handlelength=0, fontsize='large')

            
    if sup_title != None:
        plt.suptitle(sup_title)
        
    return fig, R2, bias
        

class bCNN:
    
    def __init__(self, num_features, D, arch_conv, arch_fc, 
                 arch='cnn',  ncols=1, activation=tfk.layers.ReLU(), act_mu='sigmoid', act_cov='linear', 
                 loss='chol', C=3, bn=True, maxpool=False, dropout=0.):
        self.D = D
        self.C = C
        self.arch_conv = arch_conv
        self.arch_fc   = arch_fc
        self.act = activation
        self.dropout = dropout
        self.loss = loss
        self.ncols = ncols
        
        inputs = tfk.Input(shape=(num_features, ncols), name='Spectra')
        
        if arch=='cnn':
            
            x = tfk.layers.Conv1D(filters=arch_conv[0][0], kernel_size=arch_conv[0][1],  padding='same')(inputs)
            x = activation(x)
            if bn:
                x = tfk.layers.BatchNormalization()(x)
            if maxpool:
                x = tfk.layers.MaxPool1D()(x)
            x = tfk.layers.SpatialDropout1D(dropout)(x, training=True)

            for i in arch_conv[1:]:
                x = tfk.layers.Conv1D(filters=i[0], kernel_size=i[1],  padding='same')(x)
                x = activation(x)
                if bn:
                    x = tfk.layers.BatchNormalization()(x)
                if maxpool:
                    x = tfk.layers.MaxPool1D()(x)
                x = tfk.layers.SpatialDropout1D(dropout)(x, training=True)

            x = tfk.layers.Flatten()(x)
            
        elif arch=='nn':
            x = tfk.layers.Flatten()(inputs)
            
        if len(arch_fc) > 0:
            x = tfk.layers.Dense(arch_fc[0])(x)
            x = activation(x)
            x = tfk.layers.Dropout(dropout)(x, training=True)

            for i in arch_fc[1:]:
                x = tfk.layers.Dense(i)(x)
                x = activation(x)
                x = tfk.layers.Dropout(dropout)(x, training=True)
        
        if loss == 'mse':
            outputs = tfk.layers.Dense(D)(x)
        elif loss =='chol':
            mu = tfk.layers.Dense((D), activation=act_mu)(x)
            covariance = tfk.layers.Dense(D*(D+1)/2, activation=act_cov)(x)
            outputs = tfk.layers.concatenate([mu, covariance])
            
            
        self.bnn = tfk.Model([inputs], outputs)
        
    def get_model(self):
        return self.bnn
        
    def train(self, x, y, patience=10, epochs=100, batch_size=512, lr=5e-4, ds=2000, dr=1, 
              fname='models/weights'):
        
        self.N = len(x)
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=ds,
            decay_rate=dr,
            staircase=True)
       
        opt = tfk.optimizers.Adam(lr_schedule)
        
        if self.loss == 'mse':
            self.bnn.compile(optimizer=opt, loss='mse')
        elif self.loss == 'chol':
            self.bnn.compile(optimizer=opt, loss=chol_loss(self.D))
            
        earlystop = tfk.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, mode='auto')
        
        weights_file_std = fname+'.h5'
        
        model_checkpoint =  tfk.callbacks.ModelCheckpoint(weights_file_std, monitor='val_loss', save_best_only=True,
                                                            save_weights_only=True, mode='auto',verbose=0)

        history = self.bnn.fit([x], y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1, shuffle=True, 
                  callbacks=[earlystop, model_checkpoint])

        self.bnn.summary()
        
        return history
    
    def save(self, fname):
        self.bnn.save(fname)
#             'models/' + fname + str(self.arch_conv) + '_' + str(self.arch_fc) + '_' + '_' + str(self.ncols) + '_' 
#                       + self.act + '_' + self.loss + str(self.dropout))
        
    def load_weights(self, fname):
        self.bnn.load_weights(fname)
        
    def evaluate(self, X_test, Y_test,  names, scaler=None, color_code=None, a=0.25, plot=True):
        D = self.D
        C = self.C
        
        dims_i=max(2,ceil(D/2))
        dims_j=3
        
        N = len(X_test)
        
        print('Making predictions...')
        pred_n = self.bnn.predict(X_test)
        
        mu = pred_n[:, :D]
                
        Y_test=scaler.inverse_transform(Y_test)
        mu = scaler.inverse_transform(mu)
        
        mins = Y_test.min(0)
        maxs = Y_test.max(0)
        
        R2 = [round(sk.metrics.r2_score(Y_test[:,i], mu[:,i]),2) for i in range(D)]
        
        if plot:
            print('Plotting...')
            fig, axs = plt.subplots(dims_i, dims_j, figsize=(21,6*dims_i))

            p=0
            for i in range(dims_i):
                for j in range(dims_j):
                    if p<self.D:
                        if color_code==None:
                            im=axs[i,j].scatter(Y_test[:,p],mu[:,p], alpha=a, 
                                                label = r'$R^2 = $'+str(R2[p]))#+r'; $\chi^2_{red} = $'+str(round(chisq[p],2)))
                        elif color_code== 'sigma':
                            im=axs[i,j].scatter(Y_test[:,p],mu[:,p],c=abs(As[p]),cmap="RdYlGn",
                                                vmin=0, 
                                                vmax=3,s=10, alpha=a,
                                                label = r'$R^2 = $'+str(R2[p]))#+r'; $\chi^2_{red} = $'+str(round(chisq[p],2)))
                            cbar = fig.colorbar(im, ax=axs[i,j])
                            cbar.set_label('Sigmas away from truth')
                            cbar.solids.set(alpha=1) 
                        else:
                            im=axs[i,j].scatter(Y_test[:,p],mu[:,p],c=Y_test[:,color_code],cmap="RdYlGn",
                                                vmin=min(Y_test[:,color_code]), 
                                                vmax=max(Y_test[:,color_code]),s=10, alpha=a,
                                                label = r'$R^2 = $'+str(R2[p]))#+r'; $\chi^2_{red} = $'+str(round(chisq[p],2)))
                            cbar = fig.colorbar(im, ax=axs[i,j])
                            cbar.set_label(names[color_code])
                            cbar.solids.set(alpha=1)
                        axs[i,j].plot(np.sort(Y_test[:,p]),np.sort(Y_test[:,p]),'r--')
                        axs[i,j].set_xlim([mins[p],maxs[p]])
                        axs[i,j].set_ylim([mins[p],maxs[p]])
                        axs[i,j].set_xlabel('True '+names[p], fontsize='x-large')
                        axs[i,j].set_ylabel('Predicted '+names[p], fontsize='x-large')
                        axs[i,j].grid(True)
                        axs[i,j].legend(fontsize='large')
                    else:
                        axs[i,j].remove()
                    p+=1

            plt.suptitle('R2 = ' + str(round(np.mean(R2),2)))
        
        return R2
    
    def retrieval(self, spectrum, T, noise, normalization):
        spectra = np.repeat(spectrum, T, 0)
        
        spectra_ = add_noise(spectra, noise[0], floor=noise[1])
        
        spectra = normalize(spectra_, method=normalization[0], conc=normalization[1])
        
        D = self.D
        
        pred = self.bnn.predict(spectra)
        
        if len(pred[0]) == D + D*(D+1)/2:
            mu = pred[:, :self.D]
            L  = pred[:, self.D:]
 
            L, diag = chol(L, self.D, T)
            LT = tf.transpose(L, perm=[0,2,1])
            
            Sigma = tf.linalg.inv(tf.matmul(L, LT))
            
            self.post=np.empty([T, self.D])
            for i in range(T):
                self.post[i,:] = np.random.multivariate_normal(mu[i], Sigma[i], 1)
                
        elif len(pred[0]) == D:
            self.post = pred
        
        return self.post, spectra_
    
def chol(L, D, N):
    '''
    Transforms a vector with the elements of a lower triangular matrix into a matrix.
    '''
    mat=[]
    diag=[]
    inc=0
    k = 1
    for i in range(D):
        if k == 1:
            mat1 = tf.concat( [ tf.exp(L[:,inc:inc+k]) , tf.zeros((N,D-k), dtype=tf.dtypes.float64) ], 1 )
            mat.append(mat1)
            mat=tf.reshape(mat, [N,D])
        else:
            mat1 = tf.concat( [ L[:,inc:inc+k-1], tf.exp(L[:,inc+k-1:inc+k]) , tf.zeros((N,D-k), dtype=tf.dtypes.float64) ], 1 )
            mat=tf.concat([mat, mat1],1)
        diag.append(tf.exp(L[:,inc+k-1]))
        inc+=k
        k+=1
    return tf.reshape(mat, [N,D,D]), diag

def chol_loss(D):
    '''
    Calculates the negative log-likelihood of a multivariate normal distribution.
    '''
    def loss(true, pred):
        mean = pred[:, :D]
        ll = pred[:, D:]
        N = tf.shape(true)[0]

        loss = tf.zeros((1,N))

        L, diag = chol(ll, D, N)
        LT = tf.transpose(L, perm=[0,2,1])

        S_inv=tf.matmul(L,LT)

        v = tf.cast(tf.expand_dims((true - mean),-1),dtype=tf.float64)
        vT = tf.transpose(v, perm=[0,2,1])

        inter = tf.matmul(vT,S_inv)
        quad = tf.matmul(inter, v)

        log_det = -2 * tf.cast(K.sum(K.log(diag),0), dtype=tf.float64)

        loss = log_det + quad

        return K.mean(loss)
    return loss

def load_cnn(fname):
    return tfk.models.load_model(fname)

def retrieval(obs_file, Rs, instrument, Type, N=1000, arch_conv=[(16, 17), (32, 9), (64, 7)], arch_fc=[128], 
              f_weights='../Data/CNN_weights/NIRSPEC_type1.h5', f_out='posterior.dat'):
    '''
    '''
    
    md = pickle.load(open('../Data/metadata.p', 'rb'))
    Y_scaler = pickle.load(open('../Data/yscaler_type'+str(Type)+'.p', 'rb'))
    
    try:
        obs = np.loadtxt(obs_file)
    except ValueError:
        obs = np.loadtxt(obs_file, skiprows=1)
    
    cnn = bCNN(obs.shape[0], md['D'][level], arch_conv, arch_fc,  
           arch='cnn', activation=tfk.layers.ReLU(), act_mu='sigmoid', 
           loss='chol', maxpool=True, bn=False, ncols=2, dropout=0.)
    
    cnn.load_weights(f_weights)
    
    spec = Rs**2*100*obs[:,1].reshape(1,-1)
    
    noise=1e6*obs[:,2]
    
    post_n, spectrum = cnn.retrieval(spec, N, [noise, 0.], ['-mean', 2])
    post = Y_scaler.inverse_transform(post_n)
    
    np.savetxt(f_out, post)
    
def res(wvl):
    R_ = np.empty(len(wvl))

    R_[0]  = wvl[0]/(wvl[1]-wvl[0])
    R_[-1] = wvl[-1]/(wvl[-1]-wvl[-2])

    for i in range(1, len(wvl)-1):
        R_[i] = 2*wvl[i]/(wvl[i+1]-wvl[i-1])

    return R_    

def arcis_fwd2obs(f_in, f_out, new_wvl, noise, floor=0):
    '''
    Noise in ppm
    '''
    
    trans = np.loadtxt(f_in)
    wvl = trans[:,0]
    rprs = trans[:,1]
    
    rprs = spectres(new_wvl, wvl, rprs)
    
    err = noise*np.ones(len(new_wvl))
    err[err<floor] = floor
    R = res(new_wvl)
    
    np.savetxt(f_out, np.c_[new_wvl, rprs+1e-6*err*np.random.randn(len(rprs)), 1e-6*err, R])
    
def star_spots(f_in, f_out, f_unspot, T_star, T_feature):
    h, c, k = (cte.h), (cte.c), (cte.k_B)
    
    trans = np.loadtxt(f_in)
    wvl = 1e-6*trans[:,0]
    rprs = trans[:,1]
    
    def bb(T, wvl):
        nu = c/(wvl*U.m)
        A = 2*h*nu**3/c**2
        B = 1/(np.exp(h*nu/(k*(T*U.K)))-1)
        return (A*B).value
    
    def sacf(f, bf, bs):
        return (1-f*(-bf/bs+1))**-1
    
    bf = bb(T_feature, wvl)
    bs = bb(T_star, wvl)
    
    ss = sacf(f_unspot, bf, bs)
    
    np.savetxt(f_out, np.c_[trans[:,0], ss*rprs, trans[:,2], trans[:,3]]) 