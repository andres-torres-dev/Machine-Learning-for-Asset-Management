import pandas as pd
import numpy as np

import denoisingAndDetoning

if __name__ == "__main__":
    data = pd.read_csv("data/dailyReturns.csv")
    data = data.drop(columns=["date"])

    # Punto 2:
    # compute covariance matrix
    cov = denoisingAndDetoning.covariance(data)

    # compute correlation matrix, its eigenvalues
    # and the condition number
    corr = denoisingAndDetoning.cov2corr(cov)
    eVal, eVec = denoisingAndDetoning.getPCA(corr)
    conditionNumber = denoisingAndDetoning.conditionNumber(eVal)

    print(f"Punto2:\nA. the condition number of the correlation matrix is: {conditionNumber}")

    x=np.random.normal(size=(10000,1000))
    eVal0,eVec0=denoisingAndDetoning.getPCA(np.corrcoef(x,rowvar=0))
    pdf1=denoisingAndDetoning.fitKDE(np.diag(eVal0),bWidth=.01) # empirical pdf
    print(pdf1)