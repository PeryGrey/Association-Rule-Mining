from mdlp.discretization import MDLP
from sklearn.datasets import load_iris, load_breast_cancer
import pandas

transformer = MDLP()
dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
X_disc = transformer.fit_transform(X, y)
z = pandas.DataFrame(data=X_disc, columns=dataset.feature_names)
print(X[:5])
print(z[:5])
z.to_csv('breast_cancer_discretized.csv')
