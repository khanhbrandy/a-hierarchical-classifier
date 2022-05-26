"""
Created on 2021-02-04
Creator: khanh.tn
"""


import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
from . import config

class Preprocessor:
    def __init__(self):
        self.category = pd.read_csv(config.category_name, header = 0, encoding='utf8')
        self.multi_bi_clf = config.multi_bi_clf
        self.levels = [1, 2]

    def getData(self, data_paths, label_paths):
        # Load data from sources
        print('Start getting raw data...')
        iters = zip(data_paths, label_paths)
        data_list = []
        label_list = []
        for d, l in iters:
            data = pd.read_csv(d, header = 0, encoding='utf8')
            label = pd.read_csv(l, header = 0, encoding='utf8')
            data_list.append(data)
            label_list.append(label)
        return data_list, label_list
    
    def getFeatures(self, df, feature_list):
        # Get important features
        final_df = pd.concat([df['retailer_id'], df[feature_list]],axis=1)
        return final_df

    def preprocessData(self, data, labels, level, target=None):
        start = time.time()
        print('Start processing data...')
        # Group data by category
        try:
            cat_level = 'category_level_' + str(level)
            features = self.category['Level_' + str(level) + '_kv'].unique()
        except NameError:
            print('Category level is not defined')
            
        group_df = data.groupby(by=['retailer_id',cat_level], as_index=False)['sub_GMV'].sum()
        # Pivot to get tabular data
        flat_df = group_df.pivot(index="retailer_id", columns=cat_level, values="sub_GMV").fillna(0).reset_index()
        
        # Remove unwanted columns
        norm_df = flat_df.iloc[:,1:]
        try:
            norm_df = norm_df.drop(columns=['KHONG THE PHAN LOAI'])
        except:
            pass
        # Make weighted data by row
        norm_df = norm_df.div(norm_df.sum(axis=1), axis=0)
        # Re-order and add missing features
        for f in features:
            if f not in norm_df.columns:
                norm_df[f] = 0
        norm_df = norm_df[features]
        # Get labels
        cols = list(flat_df.columns)
        
        label_df = flat_df.merge(labels, how='inner', left_on='retailer_id', right_on='retailer_id')
        # Feature encoding
        status_encode = lambda x: 1 if x =='active' else 0
        label_df['active'] = label_df['status'].map(status_encode)
        # Label encoding
        if (('corrected_industry' in labels.columns) and (target)):
            cols.extend(['status','kv_industry','corrected_industry'])
            label_df = label_df[cols]
            label_encode = lambda x: 1 if x == target else 0
            label_df['class'] = label_df['corrected_industry'].map(label_encode)
            final_df = pd.concat([label_df['retailer_id'], norm_df, label_df['class']],axis=1)
            try:
                final_df = final_df.astype({'class': int})
            except ValueError:
                print('Value error in source {}'.format(data))
        else:
            cols.extend(['status','kv_industry'])
            label_df = label_df[cols]
            final_df = pd.concat([label_df['retailer_id'], norm_df],axis=1)
        
        print('Done processing data!. Time taken = {:.1f}(s) \n'.format(time.time()-start))
        return final_df

    def concatData(self, data_list, label_list, level, target=None):
        # Combine multiple data sources
        start = time.time()
        print('Start processing data level {}...'.format(level))
        iters = zip(data_list, label_list)
        dfs = []
        for data, label in iters:
            df = self.preprocessData(data, label, level, target)
            dfs.append(df)
        final_df = pd.concat(dfs)
        
        print('Done processing data level {}. Time taken = {:.1f}(s)'.format(level, time.time()-start))
        return final_df


    def combineData(self, data_paths, label_paths, target = None):
        # Combine multiple-level data
        dfs = []
        for level in self.levels:
            data_list, label_list = self.getData(data_paths, label_paths)
            df = self.concatData(data_list, label_list, level, target)
            df = df.sort_values(by=['retailer_id'])
            dfs.append(df)
        # Transform labelled data for training process
        if target:
            label = dfs[0].iloc[:,-1]
            final_df = dfs[0].iloc[:,:-1]
            for i in range(1,len(dfs)):
                final_df = pd.concat([final_df, dfs[i].iloc[:,1:-1].reindex(final_df.index)], axis=1)
            # Select important features if given:
            if self.multi_bi_clf[target]:
                final_df = self.getFeatures(final_df, self.multi_bi_clf[target])
            final_df = pd.concat([final_df, label], axis=1)
        # Transform unlabelled data for prediction
        else:
            final_df = dfs[0]
            for i in range(1,len(dfs)):
                final_df = pd.concat([final_df, 
                                    dfs[i].iloc[:,1:].reindex(final_df.index)], axis=1)
        return final_df

if __name__ == '__main__':
    pass
    