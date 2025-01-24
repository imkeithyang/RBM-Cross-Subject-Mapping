
import numpy as np

from scipy.io import loadmat

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

########################################################################################################################################################################################################################################################################################

def DataFormatting (path, moth_database, moth, neurons, stimuli):
    
    data = loadmat(f'{path}{moth_database[0][moth]}_{moth_database[1][moth]}_AnnotatedToShare_v3.mat')
    AdditionalVars = loadmat(f'{path}{moth_database[0][moth]}_{moth_database[1][moth]}_AdditionalInfo.mat')
    
    No_trials = AdditionalVars['No_trials_perNeuronStimulus'].tolist()
    
    spike_times_ALL = [None] * len(neurons)
    targets_ALL = [None] * len(neurons)
    
    for neuron in neurons:

        spike_times_perNEURON = [None] * len(stimuli)
        targets_perNEURON = [None] * len(stimuli)

        for stimulus in stimuli:
            
            spike_times = data['{}_times'.format(neuron)][0,stimulus]
            
            if AdditionalVars['MissingData'][neurons.index(neuron),stimuli.index(stimulus)] == 0:
                
                spike_times_perNEURON[stimuli.index(stimulus)] = spike_times
                targets_perNEURON[stimuli.index(stimulus)] = stimulus*np.ones((spike_times.shape[0],1))
                
            elif AdditionalVars['MissingData'][neurons.index(neuron),stimuli.index(stimulus)] == 1:
                
                spike_times_perNEURON[stimuli.index(stimulus)] = np.empty((AdditionalVars['No_trials'][stimuli.index(stimulus),1],1))
                spike_times_perNEURON[stimuli.index(stimulus)][:] = np.NaN
                targets_perNEURON[stimuli.index(stimulus)] = stimulus*np.ones((AdditionalVars['No_trials'][stimuli.index(stimulus),1],1))
        
            #print('(Moth-{}, Neuron-{}, stim-{}) In: orig-{} / Out: targ-{}({})'.format(moth+1, neuron, stimulus, spike_times_perNEURON[stimuli.index(stimulus)].shape, targets_perNEURON[stimuli.index(stimulus)].shape, targets_perNEURON[stimuli.index(stimulus)][0,0]))

        #print(125*'-')
        spike_times_ALL[neurons.index(neuron)] = spike_times_perNEURON
        targets_ALL[neurons.index(neuron)] = targets_perNEURON
    
    return spike_times_ALL, targets_ALL, No_trials

########################################################################################################################################################################################################################################################################################

def GaussianKernelFiltering ( spike_times_ALL, targets_ALL, No_trials, neurons, stimuli, dur, f_s, sigma ):
    
    time, T = np.arange(0,dur,1/f_s), int(dur*f_s)
    gauss_kernel_response_ALL = [None] * len(neurons)
    spike_counts_ALL = [None] * len(neurons)

    for neuron in neurons:

        gauss_kernel_response_perNEURON = [None] * len(stimuli)
        spike_counts_perNEURON = [None] * len(stimuli)

        for stimulus in stimuli:

            spike_times_neuron_stimulus = spike_times_ALL[neurons.index(neuron)][stimuli.index(stimulus)]
            gauss_kernel_response = np.zeros( ( spike_times_neuron_stimulus.shape[0], T ) )
            spike_counts = np.zeros( ( spike_times_neuron_stimulus.shape[0], 1 ) )

            for trial in range(spike_times_neuron_stimulus.shape[0]):
                for i in range(spike_times_neuron_stimulus.shape[1]):
                    if ~np.isnan(spike_times_neuron_stimulus[trial,i]):
                        if spike_times_neuron_stimulus[trial,i] <= (dur*1e3):
                            gauss_kernel_response[trial,:] += np.exp( - ((time - 1e-3*spike_times_neuron_stimulus[trial,i])*(time - 1e-3*spike_times_neuron_stimulus[trial,i]))/(2*sigma*sigma) )
                            spike_counts[trial,:] += 1

            gauss_kernel_response_perNEURON[stimuli.index(stimulus)] = gauss_kernel_response
            spike_counts_perNEURON[stimuli.index(stimulus)] = spike_counts

        gauss_kernel_response_ALL[neurons.index(neuron)] = gauss_kernel_response_perNEURON
        spike_counts_ALL[neurons.index(neuron)] = spike_counts_perNEURON
    
    
    signals = np.empty((No_trials,len(neurons),T))
    signals_maxvalpos = np.empty((No_trials,len(neurons),2))
    counts = np.empty((No_trials,len(neurons),1))
    targets = np.empty((No_trials,1))

    for neuron in neurons:
        idx = 0
        for stimulus in stimuli:
            count = 0
            for trial in range( targets_ALL[neurons.index(neuron)][stimuli.index(stimulus)].shape[0] ):
                signals[idx,neurons.index(neuron),:] = gauss_kernel_response_ALL[neurons.index(neuron)][stimuli.index(stimulus)][count,:]
                signals_maxvalpos[idx,neurons.index(neuron),0] = np.argmax( gauss_kernel_response_ALL[neurons.index(neuron)][stimuli.index(stimulus)][count,:] )
                signals_maxvalpos[idx,neurons.index(neuron),1] = gauss_kernel_response_ALL[neurons.index(neuron)][stimuli.index(stimulus)][count,int(signals_maxvalpos[idx,neurons.index(neuron),0])]
                counts[idx,neurons.index(neuron),:] = spike_counts_ALL[neurons.index(neuron)][stimuli.index(stimulus)][count,:]
                targets[idx,:] = targets_ALL[neurons.index(neuron)][stimuli.index(stimulus)][count,:]
                idx, count = idx+1, count+1
    
    return signals, signals_maxvalpos, counts, targets

########################################################################################################################################################################################################################################################################################

def TrainTest_LinearDecoder( clf, signals_train=[], targets_train=[], signals_test=[], targets_test=[], mode='test' ):
    
    if mode=='train':
        clf.fit(signals_train, np.ravel(targets_train))
    
    targets_predicted = np.expand_dims(clf.predict(signals_test), axis=1)
    
    return np.sum(targets_predicted == targets_test)/targets_test.shape[0], targets_predicted, clf

########################################################################################################################################################################################################################################################################################

def train_test_split_cv ( signals, targets, train_index, test_index ):
    signals_train = signals[train_index]
    targets_train = targets[train_index]
    signals_test = signals[test_index]
    targets_test = targets[test_index]
    
    return signals_train, signals_test, targets_train, targets_test


def train_test_split_perGroup ( signals, targets, groups, train_sizes, test_sizes, random_state, cv, cv_ind ):
    
    '''
        a function that overrides the standard train_test_split function from sklearn.model_selection such that it accepts a list for the train_size and test_size arguments
        train_sizes: a list containing the number of training trials per group in groups; this is a required argument
        test_sizes: default is [None]*len(groups) in which case the function assigns the remaining trials from each group after the split as test trials; if not None, then the formatting of the argument should be equal to train_sizes
    '''
    signals_train, targets_train = np.empty(shape=(0,signals.shape[1])), np.empty(shape=(0,1))
    signals_test, targets_test = np.empty(shape=(0,signals.shape[1])), np.empty(shape=(0,1))
    
    for train_size, test_size, group in zip(train_sizes,test_sizes,groups):
        
        signals_perGroup = signals[targets.squeeze(1)==group,:]
        targets_perGroup = targets[targets.squeeze(1)==group,:]
        
        if cv:
            tot_index = np.arange(0,train_size + test_size)
            test_index = tot_index[cv_ind * test_size: (cv_ind + 1) * test_size]
            train_index = np.delete(tot_index, test_index)
            signals_train_perGroup, signals_perGroup_remain, targets_train_perGroup, targets_perGroup_remain = train_test_split_cv(signals_perGroup, targets_perGroup, train_index, test_index )
        else:
            signals_train_perGroup, signals_perGroup_remain, targets_train_perGroup, targets_perGroup_remain = train_test_split(signals_perGroup, targets_perGroup, train_size=train_size, random_state=random_state )
        signals_train = np.concatenate((signals_train,signals_train_perGroup),axis=0)
        targets_train = np.concatenate((targets_train,targets_train_perGroup),axis=0)
        
        if test_size != None and not cv:
            _, signals_test_perGroup, _, targets_test_perGroup = train_test_split(signals_perGroup_remain, targets_perGroup_remain, test_size=test_size, random_state=random_state)
        else:
            signals_test_perGroup, targets_test_perGroup = signals_perGroup_remain, targets_perGroup_remain
                                       
        signals_test = np.concatenate((signals_test,signals_test_perGroup),axis=0)
        targets_test = np.concatenate((targets_test,targets_test_perGroup),axis=0)
    
    return signals_train, signals_test, targets_train, targets_test

########################################################################################################################################################################################################################################################################################

def TrainTestSplit_VerticalSubjMerge( subjects, stimuli, signals, targets, train_sizes, test_sizes, out_dims, random_state, should_I_PCA, should_I_jointPCA, cv):
    
    signals_train, signals_test = np.empty((0,out_dims[0])), np.empty((0,out_dims[0]))
    targets_train, targets_test = np.empty((0,1)), np.empty((0,1))
    LDA_rates = np.zeros((1,len(subjects)))
    pca = PCA()
        
    for index in subjects:
                                       
        signals_train_subj, signals_test_subj, targets_train_subj, targets_test_subj = train_test_split_perGroup( signals[subjects.index(index)], targets[subjects.index(index)], stimuli, train_sizes, test_sizes, random_state )
        
        if should_I_PCA == True and should_I_jointPCA == False:
            pca = PCA(n_components=out_dims[0])
            pca.fit(signals_train_subj)
            signals_train_subj = pca.transform(signals_train_subj)
            signals_test_subj = pca.transform(signals_test_subj)
        print('moth-{}:'.format(index))
        print('       training data set size for moth-{} (tl & clf): {}'.format(index,signals_train_subj.shape))
        print('             testing  data set size for moth-{}: {}'.format(index,signals_test_subj.shape))
        
        clf = LDA()
        clf.fit(signals_train_subj,np.ravel(targets_train_subj))
        LDA_rates[0,subjects.index(index)] = np.sum(targets_test_subj == np.expand_dims(clf.predict(signals_test_subj), axis=1))/targets_test_subj.shape[0]
        print('Local LDA rate for moth-{} is {:0.3f}'.format(index, LDA_rates[0,subjects.index(index)]))
        
        
        signals_train = np.concatenate((signals_train,signals_train_subj),axis=0)
        targets_train = np.concatenate((targets_train,targets_train_subj),axis=0)
        signals_test = np.concatenate((signals_test,signals_test_subj),axis=0)
        targets_test = np.concatenate((targets_test,targets_test_subj),axis=0)
        
    if should_I_PCA == True and should_I_jointPCA == True:
        pca = PCA(n_components=out_dims[1])
        pca.fit(signals_train)
        signals_train, signals_test = pca.transform(signals_train), pca.transform(signals_test)
    
    return signals_train, signals_test, targets_train, targets_test, LDA_rates, pca

########################################################################################################################################################################################################################################################################################

def TrainTestSplit_HorizontalSubjMerge( subjects, stimuli, signals, targets, train_sizes, test_sizes, Threshold, random_state, cv, cv_ind ):
    
    signals_train, signals_test = [],[]
    targets_train, targets_test = [],[]
    P = {}
        
    for index in subjects:
                                       
        signals_train_subj, signals_test_subj, targets_train_subj, targets_test_subj = train_test_split_perGroup( signals[subjects.index(index)], targets[subjects.index(index)], stimuli, train_sizes, test_sizes, random_state, cv=cv, cv_ind=cv_ind )
        
        if Threshold <=1:
            pca = PCA()
            _ = pca.fit(signals_train_subj)
            indices = np.argwhere( np.cumsum(pca.explained_variance_ratio_) > Threshold )
            n_components = indices[0][0] + 1
        elif Threshold > 1:
            n_components = Threshold
        
        pca = PCA(n_components=(n_components))
        pca.fit(signals_train_subj)
        signals_train_subj = pca.transform(signals_train_subj)
        signals_test_subj = pca.transform(signals_test_subj)
        
        print('moth-{}:'.format(index))
        print('training data set size for moth-{} (tl & clf): {}'.format(index,signals_train_subj.shape))
        print('           testing  data set size for moth-{}: {}'.format(index,signals_test_subj.shape))
        
        P['moth-{}'.format(index)] = (signals_train_subj.shape[1], signals_train_subj.shape[1]+n_components-1)
        signals_train.append(signals_train_subj)
        targets_train.append(targets_train_subj)
        signals_test.append(signals_test_subj)
        targets_test.append(targets_test_subj)
        
    signals_train = np.concatenate(signals_train, axis=1)
    targets_train = np.concatenate(targets_train, axis=1)
    signals_test = np.concatenate(signals_test, axis=1)
    targets_test = np.concatenate(targets_test, axis=1)
    
    return signals_train, signals_test, targets_train, targets_test, P
########################################################################################################################################################################################################################################################################################

def SourceAndDestination_DataSetPreparation ( Source_DataSet, Destination_DataSet ):
    
    np.random.shuffle(Source_DataSet)
    np.random.shuffle(Destination_DataSet)
    
    X_signals, X_targets = Source_DataSet[:,:-1], Source_DataSet[:,-1]
    Y_signals, Y_targets = Destination_DataSet[:,:-1], Destination_DataSet[:,-1]
    
    X_signals_groupclass = np.empty([0,X_signals.shape[1]])
    Y_signals_groupclass = np.empty([0,Y_signals.shape[1]])
    X_targets_groupclass = np.empty([0])
    Y_targets_groupclass = np.empty([0])
    unique_targets = np.unique(X_targets) # should be equal to np.unique(Y_targets) 
    for unique_target in unique_targets:
        unique_X_target_indx = np.where(X_targets==unique_target)
        X_signals_groupclass = np.concatenate((X_signals_groupclass, X_signals[unique_X_target_indx[0].tolist()]), axis=0)
        X_targets_groupclass = np.concatenate((X_targets_groupclass, X_targets[unique_X_target_indx[0].tolist()]), axis=0)
        unique_Y_target_indx = np.where(Y_targets==unique_target)
        Y_signals_groupclass = np.concatenate((Y_signals_groupclass, Y_signals[unique_Y_target_indx[0].tolist()]), axis=0)
        Y_targets_groupclass = np.concatenate((Y_targets_groupclass, Y_targets[unique_Y_target_indx[0].tolist()]), axis=0)
        
    indx = np.arange(0, X_targets_groupclass.shape[0], 1)
    np.random.shuffle(indx)
    targets = X_targets_groupclass[indx.tolist()] # should be equal to Y_targets_groupclass[indx.tolist()]
    X_signals_groupclass = X_signals_groupclass[indx.tolist()]
    Y_signals_groupclass = Y_signals_groupclass[indx.tolist()]
    
    X_Y_signals = np.concatenate((X_signals_groupclass,Y_signals_groupclass),axis=1)
    targets = np.expand_dims(targets, axis=1)
    
    return X_Y_signals, targets

########################################################################################################################################################################################################################################################################################

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

########################################################################################################################################################################################################################################################################################

def VisualizePrincipalModes (z, signals, targets, signals_tilde, targets_tilde, colors, groups, group_names, marker, pair_of_modes, sigma, omega, epoch):
    plt.rcParams['figure.figsize'] = [6, 6]
    fig, ax = plt.subplots()
    for color, group, group_name in zip(colors, groups, group_names):
    #    ax.scatter(signals[targets.squeeze(1) == group, pair_of_modes[0]], signals[targets.squeeze(1) ==group, pair_of_modes[1]], color=color, label=group_name, alpha=1, lw=0.5, edgecolors='k', marker=marker[0])
        confidence_ellipse(signals[targets.squeeze(1) == group, pair_of_modes[0]], signals[targets.squeeze(1) ==group, pair_of_modes[1]], ax, facecolor=color, alpha=0.25)
    for color, group, group_name in zip(colors, groups, group_names):
        ax.scatter(signals_tilde[targets_tilde.squeeze(1) == group, pair_of_modes[0]], signals_tilde[targets_tilde.squeeze(1) ==group, pair_of_modes[1]], color=color, label=group_name, alpha=1, lw=1, edgecolors='k', marker=marker[1])
        #confidence_ellipse(signals_tilde[targets_tilde.squeeze(1) == group, pair_of_modes[0]], signals_tilde[targets_tilde.squeeze(1) ==group, pair_of_modes[1]], ax, edgecolor=color)
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    
    ax.set_xlabel('Mode-{}'.format(pair_of_modes[0]+1), fontsize=18)
    ax.set_ylabel('Mode-{}'.format(pair_of_modes[1]+1), fontsize=18)
    if z<=4:
        ax.set_title('Source Moth {}'.format(z), fontsize=20)
    elif z>5:
        ax.set_title('Source Moth {}'.format(z-1),fontsize=20)
    if z==10:
        plt.legend(loc='best', shadow=False, scatterpoints=1, fontsize=16)
    plt.savefig('Results/08302023/tilde_depictionTwoStrongestPmodes_Moth{}_sigma{}_omega{}_epoch{}.png'.format(z,sigma, omega, epoch))
    plt.show()
    
    return

########################################################################################################################################################################################################################################################################################

def DiscreteFourierTransform ( signals, F ):
    
    return np.dot(signals,F)

def InverseDiscreteFourierTransform ( features, F):
    
    return np.dot(features,np.linalg.pinv(F))

def FourierTransformationMatrix ( T, L, D_L=0 ):
    
    # compute Feature Extraction matrix
    if D_L == 0:
        F = np.ones([1,T])
        for i in range(1,L):
            F = np.concatenate((F,np.expand_dims((2**(1/2))*np.cos(np.pi*2*i*(np.arange(0, T, 1)/T)),axis=0)),axis=0)
            F = np.concatenate((F,np.expand_dims((2**(1/2))*np.sin(np.pi*2*i*(np.arange(0, T, 1)/T)),axis=0)),axis=0)
    else:
        F = np.empty([0,T])
        for i in range(D_L,D_L+L):
            F = np.concatenate((F,np.expand_dims((2**(1/2))*np.cos(np.pi*2*i*(np.arange(0, T, 1)/T)),axis=0)),axis=0)
            F = np.concatenate((F,np.expand_dims((2**(1/2))*np.sin(np.pi*2*i*(np.arange(0, T, 1)/T)),axis=0)),axis=0)
    F = (1/T)*np.transpose(F,(1,0))
    
    return F

def ExtractFeatures ( signals, L, D_L=0 ): # signals.shape -> ( #trials, N, T )
    
    F = FourierTransformationMatrix (signals.shape[2],L,D_L)
    
    # compute DFT coefficients (i.e., extract features)
    features = DiscreteFourierTransform(signals,F)
    signals_reconst = InverseDiscreteFourierTransform(features,F)
    
    return signals, signals_reconst, features
