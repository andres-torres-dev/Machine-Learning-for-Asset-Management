import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity

# getPCA computes the eigenvalues and
# eigenvector from given matrix
def getPCA(matrix):
    eVal, eVec = np.linalg.eigh(matrix)
    
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:,indices]

    eVal = np.diagflat(eVal)
    
    return eVal, eVec

def mpPDF(var, q, pts=1000):
    # Marcenko-Pastur pdf
    # q=T/N
    eMin, eMax = var*(1-(1./q)**.5)**2, var*(1+(1./q)**.5)**2
    eVal = np.linspace(eMin,eMax,pts)
    pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf = pd.Series(pdf, index=eVal)
    
    return pdf

def fitKDE(obs, bWidth=.25, kernel="gaussian", x=None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape)==1:obs=obs.reshape(-1,1)
    kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
    if x is None:x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1:x=x.reshape(-1,1)
    logProb=kde.score_samples(x) # log(density)
    pdf=pd.Series(np.exp(logProb),index=x.flatten())

    return pdf

def covariance(matrix):
    return np.cov(matrix)

def cov2corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1 # fix numerical error

    return corr

def corr2cov(corr,std):
    cov = corr * np.outer(std, std)
    
    return cov

# conditionNumber takes the eigevalues
# of the correlation matrix and returns their ratio
def conditionNumber(eVal):
    eVal = np.diag(eVal)

    return eVal[1] / eVal[0]

# denoiseCorrelation implements the constant residual eigenvalue method
# to reduce the noise of the data while preserving the data
def denoiseCorrelation(eVal, eVec, nFacts):
    eVal[nFacts:] = eVal[nFacts:].sum() / float(eVal.shape[0] - nFacts)
    eVal = np.diag(eVal)
    cov = np.dot(eVec, eVal).dot(eVec.T)

    return cov2corr(cov)

# detoneCorrelation removes the market component from a
# denoised correlation matrix eigenvalues and eigenvectors 
# where nFacts respresent the number of components 
# associated with market noise
def detoneCorrelation( eVal, eVec, nFacts):
    c2_ = np.dot(eVec[:, nFacts: ], eVal[:, nFacts:]).dot(eVec[:, nFacts:].T)
    c2 = c2_.dot(np.linalg.inv(np.dot(np.sqrt(np.diag(c2_)), np.sqrt(np.diag(c2_).T))))

    return c2



