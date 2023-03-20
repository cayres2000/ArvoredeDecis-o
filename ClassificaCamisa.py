#pip install scikit-learn==1.0.2

#Rodrigo de Souza Garcia R.A.:22.120.034-8
#Vinicius Cayres Gago R.A.:22.120.034-8


import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff

def criterioProvas():
    data,meta = arff.loadarff('./Atividade/ClassificaCamisa.arff')

    attributes = meta.names()
    data_value = np.asarray(data)

    altura = np.asarray(data['altura']).reshape(-1,1)
    torax = np.asarray(data['torax']).reshape(-1,1)
    features = np.concatenate((altura, torax),axis=1)
    target = data['tamanho']


    Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

    plt.figure(figsize=(10, 6.5))
    tree.plot_tree(Arvore,feature_names=['alturax','torax'],class_names=['P','M','G'],
                   filled=True, rounded=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(25, 10))
    metrics.plot_confusion_matrix(Arvore,features,target,display_labels=['P','M','G'], values_format='d', ax=ax)
    plt.show()

criterioProvas()