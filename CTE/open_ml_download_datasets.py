#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 23:19:14 2024

@author: xbb
"""
import numpy as np
import openml

if __name__ == '__main__':
    for j in range(19):        
        SUITE_ID = 336 # Regression on numerical features        
        tasks = openml.study.get_suite(SUITE_ID).tasks  
        
        # 0 : cpu_act
        # 1 : pol
        # 2 : elevators
        # 3 : wine_quality
        # 4 : Ailerons
        # 5 : houses
        # 6 : house_16H
        # 7 : diamonds
        # 8 : Brazilian_houses
        # 9 : Bike_Sharing_Demand
        # 10 ; nyc-taxi-green-dec-2016
        # 11 : house_sales
        # 12 : sulfur
        # 13 : medical_charges
        # 14 : MiamiHousing2016
        # 15 : superconduct
        # 16 : yprop_4_1
        # 17 : abalone
        # 18 : delay_zurich_transport
        
        
        task = openml.tasks.get_task(tasks[j]) # download the OpenML task
        dataset = task.get_dataset()
        
        print(f"The content of the {j}th dataset: {dataset}")
        print("You may save the datasets by `pickle.dump` function")
        
        X, y, categorical_indicator, attribute_names \
            = dataset.get_data(dataset_format = "dataframe", 
            target = dataset.default_target_attribute)        
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        
        