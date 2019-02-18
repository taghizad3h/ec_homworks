import logging

from numpy import genfromtxt
import numpy as np


def get_first_dataset(path='data/Concrete_Data.csv'):
    my_data = genfromtxt(path, delimiter=',')
    X = my_data[:, :-1]
    y = my_data[:, -1]
    var_names = ['cement','blastFurnaceSlag','flyAsh','water','Superplasticizer','coarseAggregate','fineAggregate','age']
    return X, y, var_names