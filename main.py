# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:14:57 2022

@author: felima
"""

import pathlib
import functions
import pickle

dataset_path = "/home/rodriguez/Documents/FaceMoCap_ML_Project/Data FaceMoCap/Sujets Sains"

ref_csv =  sorted(str(p) for p in pathlib.Path(dataset_path).glob("*.csv"))

list_ref = functions.create_list_ref(ref_csv)

list_dataset = functions.create_list_dataset(list_ref)

list_ds_interpolated = functions.interpolate_list(list_dataset)

list_ds_int_fix_dataset = functions.create_fixed_duration_dataset(list_ds_interpolated, fixed_duration=500)

list_ds_int_fix_dental = functions.dental_frame(list_ds_int_fix_dataset)

list_ds_int_fix_dental_disp = functions.displacement_list(list_ds_int_fix_dental)

# add normalized_displacement

with open('list_ds_int_fix_dental_disp.pkl', 'wb') as file:
    pickle.dump(list_ds_int_fix_dental_disp, file)

