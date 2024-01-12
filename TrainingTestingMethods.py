import torch
import torch.nn.functional as func
import numpy as np

def TrLeCVAE_train ( model, Data, optimizer, alpha ) :
    model.train()
    for i, batch in enumerate(Data):
        train_In, train_Out = batch[:,:model.CVAE_in_dim], batch[:,model.CVAE_in_dim:]
        optimizer.zero_grad()
        recog_train_Out_tilde, logvars_recog_train_Out, prior_train_Out_tilde, logvars_prior_train_Out, Zr_mu, Zr_logvar, Zp_mu, Zp_logvar = model( train_In, train_Out )
        recog_LL = func.smooth_l1_loss(recog_train_Out_tilde, train_Out, reduction='sum')
        #recog_LL = torch.sum( logvars_recog_train_Out ) + func.smooth_l1_loss(recog_train_Out_tilde*(logvars_recog_train_Out.exp().pow(-0.5)), train_Out*(logvars_recog_train_Out.exp().pow(-0.5)), reduction='sum')
        #prior_LL = torch.sum( logvars_prior_train_Out ) + func.smooth_l1_loss(prior_train_Out_tilde*(logvars_prior_train_Out.exp().pow(-0.5)), train_Out*(logvars_prior_train_Out.exp().pow(-0.5)), reduction='sum')
        D_KL = -torch.sum( 1 + Zr_logvar - Zp_logvar - (Zr_logvar.exp()+(Zr_mu-Zp_mu).pow(2))/(Zp_logvar.exp()) )
        loss = D_KL + recog_LL #alpha*(D_KL + recog_LL) + (1-alpha)*prior_LL # D_KL + recog_LL
        loss.backward()
        optimizer.step()
    return model

def TrLeCVAE_test ( model, TestingSignals, TestingSignals_dummy ) :
    model.eval()
    with torch.no_grad():
        TestingSignals_tilde_mu, TestingSignals_tilde_logvar = model( TestingSignals, TestingSignals_dummy )
        eps = torch.randn_like( TestingSignals_tilde_mu )
        TestingSignals_tilde = TestingSignals_tilde_mu #eps.mul( torch.exp(TestingSignals_tilde_logvar/2) ).add_( TestingSignals_tilde_mu )
    return TestingSignals_tilde

def train_RBM_forTL ( model, TrainingData, optimizer, flag='CD', k=1 ) :
    model.train()
    for i, batch in enumerate(TrainingData) :
        optimizer.zero_grad()
        if flag=='CD':
            vh_k,_ = model(batch, k=k)
            loss = model.free_energy(batch) - model.free_energy(vh_k)
        elif flag=='FD':
            loss = model.fisher_score(batch)
        loss.backward()
        optimizer.step()
    return model

def test_RBM_forTL ( model, TestingData, k ):
    model.eval()
    with torch.no_grad():
        vh_k, pred = model(TestingData, k=k)
    return vh_k, pred


def CrossSubjectDecoding_eval (TestingSignals_tilde, X_targets_test, clf_Y, clf_type) : 
    if clf_type == 'LDA':
        rate = 0
        signals_test_tilde = TestingSignals_tilde.detach().cpu().numpy().astype('float64')
        targets_test_tilde = clf_Y.predict(signals_test_tilde)
        rate = np.sum(targets_test_tilde==np.ravel(X_targets_test))/X_targets_test.shape[0]
        
    elif clf_type == 'MLP':
        rate = 0
        scores = clf_Y( TestingSignals_tilde )
        _, test_Out_tilde = torch.max(scores, 1)
        targets_test_tilde = test_Out_tilde.cpu().numpy().astype('float64')
        rate = np.sum(targets_test_tilde==np.ravel(X_targets_test))/X_targets_test.shape[0]
        
    return rate

def ClfMLP_train ( model, TrainingData, optimizer ) :
    model.train()
    for i, batch in enumerate(TrainingData):
        train_In, train_Out = batch[:,:-1], batch[:,-1]
        optimizer.zero_grad()
        train_scores = model( train_In )
        loss = func.cross_entropy( train_scores, torch.autograd.Variable(train_Out.long()) )
        loss.backward()
        optimizer.step()
    return model

def ClfMLP_eval ( model, TestingData ) :
    model.eval()
    rate = 0
    with torch.no_grad():
        for _, batch in enumerate(TestingData):
            test_In, test_Out = batch[:,:-1], batch[:,-1]
            scores = model( test_In )
            _, test_Out_tilde = torch.max(scores, 1)
            rate = ( torch.sum( torch.eq( test_Out.long(), test_Out_tilde ) ).item() )/test_Out.size(0)  
    return rate


