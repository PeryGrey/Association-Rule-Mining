from mdlp.discretization import MDLP
from sklearn.datasets import load_iris, load_breast_cancer
import pandas as pd

z = pd.read_csv('adult.csv')
transformer = MDLP()
X, y = z, z['income']
X.drop(['workclass', 'education', 'marital.status', 'occupation', 'relationship',
        'race', 'sex', 'native.country', 'income'], inplace=True, axis=1)
ar = ['clump_thickness', 'size_uniformity', 'shape_uniformity', 'marginal_adhesion',
      'epithelial_size', 'bare_nucleoli', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
ar1 = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
ar2 = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
ar3 = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
ar4 = ['age', 'fnlwgt', 'education.num',
       'capital.gain', 'capital.loss', 'hours.per.week']
X_disc = transformer.fit_transform(X, y)
# print(X_disc)

b = pd.DataFrame(data=X_disc, columns=ar4)
b.to_csv('discretized-adult.csv')
