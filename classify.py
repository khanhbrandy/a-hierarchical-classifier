"""
Created on 2021-02-04
Creator: khanh.tn
"""

import pandas as pd
import numpy as np
import time
import os
import joblib
import multiprocessing
from multiprocessing import Pool
from functools import partial
from source.modules.preprocess import Preprocessor
from source.modules import config

class Classifier:
    def __init__(self):
        self.preprocessor = Preprocessor()
    def loadModel(self):
        clfs = {}
        for target in self.preprocessor.multi_bi_clf.keys():
            clfs[target] = joblib.load(os.path.join(os.getcwd(), config.model_dir, target, config.model_name))
        return clfs

    def loadData(self, data_paths, label_paths, target=None):
        final_df = self.preprocessor.combineData(data_paths, label_paths, target)
        return final_df

    def classify(self, index, df, clfs):
        # Init prediction
        pred = 'Ngành khác'
        for target in clfs.keys():
            df_n = self.preprocessor.getFeatures(df, self.preprocessor.multi_bi_clf[target])
            X = df_n.iloc[index, 1:]
            # Convert from Pandas Series to Numpy Array and reshape
            X = np.array(X).reshape((1,-1))
            if clfs[target].predict(X) == 1:
                pred = target
                break
            else:
                continue
        return pred

def main(data_paths, label_paths):
    print('Start predicting...')
    start = time.time()
    # Init models
    classifier = Classifier()
    clfs = classifier.loadModel()
    # Get data
    df = classifier.loadData(data_paths, label_paths)
    # Multi-predict
    indices = list(range(df.shape[0]))
    n_processes = multiprocessing.cpu_count() * 2 + 1
    pool = Pool(n_processes)
    result = pool.map(partial(classifier.classify, df=df, clfs=clfs), indices)
    pool.close()
    pool.join()
    df['bi_industry'] = result
    print('Done predicting data!. Time taken = {:.1f}(s) \n'.format(time.time()-start))
    return df

if __name__ == '__main__':
    data_paths = [config.validating_data_1]
    label_paths = [config.validating_label_1]
    df = main(data_paths, label_paths)
    #  Save results, should be removed out when going live
    filename = config.validating_label_1.replace('.csv', '_preds.csv')
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    


    
    
   