# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle as pkl

def pkl_dump(data, file, output_path):
    with open(output_path + file, "wb") as fw:
        pkl.dump(data, fw, pkl.HIGHEST_PROTOCOL)

def pkl_load(file, input_path):
    with open(input_path + file, "rb") as fr:
        data = pkl.load(fr)
    return data