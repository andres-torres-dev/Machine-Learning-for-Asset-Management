#VARIATION INFORMATION
import numpy as np, scipy.stats as ss
from sklearn.metrics import mutual_info_score

def varInfo(x,y,bins,norm=False):
    cXY=np.histogram2d(x,y,bins) [0]
    iXY=mutual_info_score(None,None,contingency=cXY)
    hX=ss.entropy(np.histogram(x,bins)[0]) #marginal
    hY=ss.entropy(np.histogram(y.bins)[0]) #marginal
    vXY=hX+hY-2*iXY #variation of information
    if norm:
        hXY=hX+hY-iXY #joint
        vXY/=hXY #normalized variation of information
    return vXY

#VARIATION OF INFORMATION ON DISCRETIZED CONTINUOS RANDOM VARIABLES
#Find optimal number of bins
def numBins(nObs, corr=None):
    if corr is None: #univariate case
        z=(8+324*nObs+12*(36*nObs+729*nObs**2))**.5)**(1/3.)
    else: #bivariate case
        b=round(2**-.5*(1+(1+24*nObs/(1.-corr**2))**.5)**.5)
    return int(b)

def varInfo(x,y,norm=False):
    bXY=numBins(x.shape[0], corr=np.corrcoef(x,y)[0,1])
    cXY=np.histogram2d(x,y,bXY) [0]
    iXY=mutual_info_score(None,None,contingency=cXY)
    hX=ss.entropy(np.histogram(x,bXY)[0]) #marginal
    hY=ss.entropy(np.histogram(y.bXY)[0]) #marginal
    vXY=hx+hY-2*iXY #variation of information
    if norm:
        hXY=hX+hY-iXY #joint
        vXY/=hXY #normalized variation of information
    return vXY

#CORRELATION AND NORMALIZED MUTUAL INFORMATION OF TWO INDEPENDENT GAUSSIAN RANDOM VARIABLES
#Mutual information
def mutualInfo(x,y,norm=False):
    bXY=numBins(x.shape[0], corr=np.corrcoef(x,y)[0,1])
    cXY=np.histogram2d(x,y,bXY) [0]
    iXY=mutual_info_score(None,None,contingency=cXY)
    if norm:
        hX=ss.entropy(np.histogram(x,bXY)[0])
        hY=ss.entropy(np.histogram(y.bXY)[0])
        iXY/=min(hX,hY) #normalized mutual information
    return iXY
    


