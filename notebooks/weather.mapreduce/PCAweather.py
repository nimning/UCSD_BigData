import re,pickle,base64,zlib
from mrjob.job import MRJob
import pandas as pd
import numpy as np
import sklearn as sk
import random

labeled_Djoined = pd.read_csv('labeled_Djoined.csv')

class PCAWEATHER(MRJob):

    def mapper(self, _, line):
        elements = line.split(',')
        stationinfo = labeled_Djoined[labeled_Djoined['station'] == elements[0]]
        #can find the station and TMAX measurment
        if len(stationinfo) != 0 and elements[1] == 'TMAX':
            #issues here?????
            leaflabel = stationinfo.iloc[0]['leaflabel']
            yeardata = pd.DataFrame(elements[3:])
            #print yeardata
            yeardata[yeardata == ''] = float('NaN')
            yeardata = yeardata.astype(float)
            mean = np.mean(yeardata)
            row = yeardata - mean
            outer = np.outer(row, row).tolist()
            yield leaflabel, outer
    
    def reducer(self, leaflabel, vect_covmatrices):
        covmatrices = list(vect_covmatrices)
        #NaN-tolerant averaging
        C = np.zeros(np.shape(covmatrices[0]))
        N = np.zeros(np.shape(covmatrices[0]))
        for covmatrix in covmatrices:
            outer = np.array(covmatrix)
            valid = np.isnan(outer) == False
            C[valid] = C[valid] + outer[valid]
            N[valid] = N[valid] + 1    
        valid_outer = np.multiply(1-np.isnan(N), N>0)
        cov = np.divide(C, N)
        cov = np.multiply(cov, valid_outer)
        U, D, V = np.linalg.svd(cov)
        cum_sum = np.cumsum(D[:])/sum(D)
        for i in range(len(cum_sum)):
            if cum_sum[i] >= 0.95:
                num_valideig = i 
                break
        yield leaflabel, num_valideig

if __name__ == '__main__':
    PCAWEATHER.run()