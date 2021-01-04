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

    pdf0 = denoisingAndDetoning.mpPDF(data.to_numpy()) # theorical pdf
    pdf1 = denoisingAndDetoning.fitKDE(np.diag(eVal)) # empirical pdf
    
    print(np.sum((pdf0 - pdf1) **2))