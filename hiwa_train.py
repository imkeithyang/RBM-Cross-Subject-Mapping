# We follow feature space notation: Subject A is denoted as Subject X (source), Subject B is denoted as Subjecy Y (destination)
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.io import loadmat
from scipy.io import savemat
import pickle
from hiwa_np import HiWA
from scipy.linalg import sqrtm, orth
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import Data Extraction and Formatting FUnctions
from MothDataFormattingOtherFunctions import DataFormatting
from MothDataFormattingOtherFunctions import GaussianKernelFiltering
from MothDataFormattingOtherFunctions import TrainTest_LinearDecoder
from MothDataFormattingOtherFunctions import SourceAndDestination_DataSetPreparation
from MothDataFormattingOtherFunctions import TrainTestSplit_HorizontalSubjMerge
from MothDataFormattingOtherFunctions import VisualizePrincipalModes

# Import Models and Methods
from Models import ConditionalVAE as CVAE
from Models import MultiLayerPerceptron as MLP
from Models import GaussBernRBM_withUnitVariances as GBRBM__withUnitVariances
from Models import GaussBernRBM_withNonUnitVariances as GBRBM__withNonUnitVariances
from TrainingTestingMethods import TrLeCVAE_train
from TrainingTestingMethods import TrLeCVAE_test
from TrainingTestingMethods import train_RBM_forTL
from TrainingTestingMethods import test_RBM_forTL
from TrainingTestingMethods import CrossSubjectDecoding_eval
from TrainingTestingMethods import ClfMLP_train
from TrainingTestingMethods import ClfMLP_eval

"""Get parser object."""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(
    description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-z_eval",
    "--z_eval",
    dest="z_eval",
    default="0",
    help="Z to eval",
    required=True,
)

parser.add_argument(
    "-d",
    "--datapath",
    dest="datapath",
    default="",
    help="Data Path",
    required=True,
)

args = parser.parse_args()
eval_index = args.z_eval

datapath = args.datapath

x = loadmat(datapath + 'Moths/Moth10_20201013_AnnotatedToShare_v3.mat')
x.keys()

# Somewhat global variables (never change)
path = "/hpc/group/tarokhlab/hy190/data/Moths/Moth"
index_database = [ [1       , 2       , 3       , 4       , 5       , 6       , 7       , 8       , 9       ,       10], 
                   [20200113, 20200114, 20200219, 20200221, 20200226, 20200818, 20200819, 20200915, 20201006, 20201013] ]
missing_data_indices_database = [ [2-1,10-1], [], [6-1], [7-1], [2-1,6-1], [], [9-1], [], [], [6-1,10-1] ]
all_neurons = ['LAX','LBA','LDLM','LDVM','LSA',  'RAX','RBA','RDLM','RDVM','RSA']
forces = ['Fx_by_WS','Fy_by_WS','Fz_by_WS','Tx_by_WS','Ty_by_WS','Tz_by_WS']

stimuli = [0,1,2,3,4,5]
colors = ['orange','red','gold','limegreen','dodgerblue','darkviolet']
stimuli_names = ['pitch up','pitch down','roll left','roll right','yaw left','yaw right']

# These are also global variables (change in different studies)
fs, duration = 10e3, 0.06
T = int(fs*duration)
N = 10
sigmas = [0.0025]

MCT = 100
X_Seeds = list(range(MCT))
E, batch_size = 350, 150
learning_rates = {}
learning_rates['both_MLandF'] = [.5e-2]
step_size, gamma, weight_decay = 1, 1, 0
init_tol = 1e-1

Thresholds = [ 10 ]
no_HiddenUnits = [ 15 ]
omegas = [ 0.5 ]

Z = [1,2,3,4,6,7,8,9,10]

temp_F = {}
temp_ML = {}
acc_all = []
for index in Z:
    temp_F['moth-{}'.format(index)] = []
    temp_ML['moth-{}'.format(index)] = []

hiwa_stat_all = []

for sigma in sigmas:

    signals_Z, targets_Z, no_Trials_Z = [None]*len(Z), [None]*len(Z), [1000000]*len(stimuli)
    for index in Z:
        spike_times, targets, no_Trials = DataFormatting( path, index_database, index-1, all_neurons, stimuli )
        signals, _, _, targets = GaussianKernelFiltering( spike_times, targets, sum(no_Trials[0]), all_neurons, stimuli, duration, fs, sigma )
        signals = signals.reshape((signals.shape[0],signals.shape[1]*signals.shape[2]))
        signals_Z[Z.index(index)], targets_Z[Z.index(index)], no_Trials_Z = signals, targets, np.minimum(no_Trials_Z,no_Trials[0])

    for omega in omegas:

        train_sizes = (omega*no_Trials_Z).astype('int').tolist()
        test_sizes = (0.9*(1-omega)*no_Trials_Z).astype('int').tolist()

        for Threshold in Thresholds:

            for h_dim in no_HiddenUnits:
                
                Rate_LDA, Rate_MLRBM, Rate_FRBM, Rate_NO, Rate_HiWA = {}, {}, {}, {}, {}
                for index in Z:
                    Rate_LDA['moth_{}'.format(index)] = np.zeros((MCT))
                    Rate_MLRBM['moth_{}'.format(index)] = np.zeros((E, MCT))
                    Rate_FRBM['moth_{}'.format(index)] = np.zeros((E, MCT))
                    Rate_NO['moth_{}'.format(index)] = np.zeros((MCT))
                    Rate_HiWA['moth_{}'.format(index)] = np.zeros((MCT))
                
                for lr in learning_rates['both_MLandF']:
                
                    for m in range(MCT):

                        signals_Z_train, signals_Z_test, targets_Z_train, targets_Z_test, P_Z = TrainTestSplit_HorizontalSubjMerge(Z, stimuli, signals_Z, targets_Z, train_sizes, test_sizes, Threshold, m)
                        print(100*'-')
                        print('Training Data shapes: {}, {}'.format(signals_Z_train.shape, targets_Z_train.shape)) 
                        print(' Testing Data shapes: {}, {}'.format(signals_Z_test.shape, targets_Z_test.shape)) 
                        print(100*'-')

                        clfs_Z = {}
                        for index in Z:
                            clfs_Z['clf_moth-{}'.format(index)] = LDA()
                            Rate_LDA['moth_{}'.format(index)][m], _, clfs_Z['clf_moth-{}'.format(index)] = TrainTest_LinearDecoder( clfs_Z['clf_moth-{}'.format(index)], 
                                                                                                                                   signals_Z_train[:,P_Z['moth-{}'.format(index)][0]:P_Z['moth-{}'.format(index)][1]+1], 
                                                                                                                                   np.expand_dims(targets_Z_train[:,0],axis=1), signals_Z_test[:,P_Z['moth-{}'.format(index)][0]:P_Z['moth-{}'.format(index)][1]+1], np.expand_dims(targets_Z_test[:,0],axis=1), mode='train' )
                            print('  LDA rate for moth-{}: {:0.4f}'.format(index,Rate_LDA['moth_{}'.format(index)][m]))
                        print(100*'-')

                        hiwa_list=[]
                        for index in Z:
                            if index != int(eval_index): continue
                            X_Z_train_HiWA = signals_Z_train[:,P_Z['moth-{}'.format(index)][0]:P_Z['moth-{}'.format(index)][1]+1]
                            X_Z_train_label = targets_Z_train[:,Z.index(index)]
                            signals_Z_test_init_hiWA_list = []
                            acc_list = []
                            pred_list = []
                            print(eval_index)
                            for index_hiwa, moth_index in enumerate(Z):
                                if index == moth_index: continue
                                # HiWA requires normalization to prevent numeric issue
                                hwa = HiWA(dim_red_method=PCA(n_components=10), normalize=False)
                                Y_Z_train_HiWA = signals_Z_train[:,P_Z['moth-{}'.format(moth_index)][0]:P_Z['moth-{}'.format(moth_index)][1]+1]
                                Y_Z_test_HiWA = signals_Z_test[:,P_Z['moth-{}'.format(moth_index)][0]:P_Z['moth-{}'.format(moth_index)][1]+1]

                                Y_Z_train_HiWA_normal = (Y_Z_train_HiWA - Y_Z_train_HiWA.mean(0))/(Y_Z_train_HiWA.std(0))
                                Y_Z_test_HiWA_normal = (Y_Z_test_HiWA - Y_Z_train_HiWA.mean(0))/(Y_Z_train_HiWA.std(0))
                                Y_Z_train_label = targets_Z_train[:,Z.index(moth_index)]
                                Y_Z_test_label = targets_Z_test[:,Z.index(moth_index)]

                                X_Z_train_HiWA = signals_Z_train[:,P_Z['moth-{}'.format(index)][0]:P_Z['moth-{}'.format(index)][1]+1]
                                X_Z_train_HiWA_normal = (X_Z_train_HiWA - X_Z_train_HiWA.mean(0))/(X_Z_train_HiWA.std(0))
                                X_Z_train_label = targets_Z_train[:,Z.index(index)]

                                hwa.fit(Y_Z_test_HiWA_normal, Y_Z_test_label, X_Z_train_HiWA_normal, X_Z_train_label)
                                temp = hwa.transform(Y_Z_test_HiWA_normal)
                                Y_Z_test_HiWA_pred_unormal = temp*(Y_Z_train_HiWA.std(0)) + Y_Z_train_HiWA.mean(0)
                                acc, pred, _ = TrainTest_LinearDecoder ( clf=clfs_Z['clf_moth-{}'.format(index)], 
                                                                        signals_test=Y_Z_test_HiWA_pred_unormal, 
                                                                        targets_test=Y_Z_test_label.reshape(-1,1), mode='test' )
                                acc_list.append(acc)
                                pred_list.append(pred)
                                print("Seed: {}, Target: {}, Source: {}, acc: {}".format(m, index, moth_index, acc))
                                # unnormalize the data (transformed to X's space, so unnormalize with X's normalization)
                                signals_Z_test_init_hiWA_list.append(Y_Z_test_HiWA_pred_unormal)
                                hiwa_list.append(hwa)
                                
                            signals_Z_test_init_hiWA = np.array(signals_Z_test_init_hiWA_list).reshape((-1,Threshold))
                            targets_Z_test_init_hiWA = np.delete( targets_Z_test.copy(), Z.index(index), 1 ).reshape((-1,1))
                            # Code to transform signals_Z_test_init using some other methods. 

                            Rate_HiWA['moth_{}'.format(index)][m], _, _ = TrainTest_LinearDecoder ( clf=clfs_Z['clf_moth-{}'.format(index)], 
                                                                                                signals_test=signals_Z_test_init_hiWA, 
                                                                                                targets_test=targets_Z_test_init_hiWA, mode='test' )

                        savemat('Results/08302023/HiWA_SCENARIO_I_CD_OneMissingSubj_sigma{}_PCAThreshold{}_noHidUnit{}_lr{}_tol{}_TrainTestRatio{}.mat'.format(sigma,Threshold,h_dim,lr,init_tol,omega),
                            {'X_Seeds':X_Seeds,'MCT':MCT,
                            'fs':fs,'duration':duration,
                            'E':E,'batch_size':batch_size,'learning_rates':learning_rates,'weight_decay':weight_decay,'step_size':step_size,'gamma':gamma,
                            'Rate_LDA':Rate_LDA,'Rate_MLRBM':Rate_MLRBM,'Rate_NO':Rate_NO})
                        savemat('Results/08302023/HiWA_SCENARIO_I_F_OneMissingSubj_sigma{}_PCAThreshold{}_noHidUnit{}_lr{}_tol{}_TrainTestRatio{}.mat'.format(sigma,Threshold,h_dim,lr,init_tol,omega),
                                {'X_Seeds':X_Seeds,'MCT':MCT,
                                'fs':fs,'duration':duration,
                                'E':E,'batch_size':batch_size,'learning_rates':learning_rates,'weight_decay':weight_decay,'step_size':step_size,'gamma':gamma,
                                'Rate_LDA':Rate_LDA,'Rate_FRBM':Rate_FRBM,'Rate_NO':Rate_NO})
                            

