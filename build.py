"""
Created on 2021-02-04
Creator: khanh.tn
"""

import pandas as pd
import numpy as np
import time
from source.modules.model import Model
from source.modules.preprocess import Preprocessor
from source.modules import config
import os
import shutil
import joblib
from copy import copy as make_copy


def build(data_paths, label_paths):
    model = Model()
    preprocessor = Preprocessor()
    for target in preprocessor.multi_bi_clf.keys():
        clf = make_copy(model.xgb_clf)
        final_df = preprocessor.combineData(data_paths, label_paths, target)
        # Train/Test split
        X, y, X_train, X_test, y_train, y_test = model.train_test_split(clf, final_df)
        model.train(clf, X_train, y_train, X_test, y_test)
        model.cross_validate(clf, X, y)
        # Store predictions
        save_dir = os.path.join(os.getcwd(), config.model_dir, target)
        save_path = os.path.join(save_dir,target+'_predictions.csv')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        y_pred, y_pred_proba = model.predict(clf, X)
        final_df['pred_proba'] = y_pred_proba
        final_df['pred'] = y_pred
        final_df.to_csv(save_path, index=False, encoding='utf8')
        # Dump trained model
        joblib.dump(clf, os.path.join(save_dir, config.model_name))
        print('Result and model for {} saved! \n'.format(target))
    pass

def transform(data_paths, label_paths, target=None):
     preprocessor = Preprocessor()
     final_df = preprocessor.combineData(data_paths, label_paths, target)
     final_df.to_csv('final_df.csv', index=False, encoding='utf-8-sig')
     pass


if __name__ == '__main__':
    data_paths = [
                    config.training_data_1,
                    config.training_data_2,
                    config.training_data_3,
                    config.training_data_4,
                    config.training_data_5,
                    config.training_data_6,
                    config.training_data_7,
                    config.training_data_8,
                    config.training_data_9,
                    config.training_data_10,
                    config.training_data_11,
                    config.training_data_12,
                    config.training_data_13,
                    config.training_data_14,
                    ]
    label_paths = [
                    config.training_label_1,
                    config.training_label_2,
                    config.training_label_3,
                    config.training_label_4,
                    config.training_label_5,
                    config.training_label_6,
                    config.training_label_7,
                    config.training_label_8,
                    config.training_label_9,
                    config.training_label_10,
                    config.training_label_11,
                    config.training_label_12,
                    config.training_label_13,
                    config.training_label_14,
                    ]
    build(data_paths = data_paths, label_paths = label_paths)
    # transform(data_paths = data_paths, label_paths = label_paths)

    