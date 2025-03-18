# easyELM
easy ELM is a Python Implementation of Extreme Learning Machines.

Extreme learning machine is a new learning algorithm for single hidden layer feedforward neural networks (SLFN). ELM generates the input to hidden layer weights randomly by any probability distribution and analytically determines the output weights. 

ELM is much more faster in training than gradient descendent algorithms, it has better generalization and provides similar accuracy to other popular methods such as gradient descent Multilayer Perceptron and Support Vector Machines.

How to use
--------------------------------------------------------------------------------
easyELM uses a scikit-learn like syntax:

```
from sklearn import datasets
from sklearn.model_selection import train_test_split
from easyELM import ELMCLassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

elmc = ELMCLassifier(hidden_layer_neurons=10, activation='sigmoid')
elmc.fit(X_train, y_train)
predictions = elmc.predict(X_test)
print('Test accuracy: %f' % elmc.score(X_test, y_test))
```

References
--------------------------------------------------------------------------------
Huang, Guang-Bin & Zhu, Qin-Yu & Siew, Chee. (2004). Extreme learning machine: A new learning scheme of feedforward neural networks. IEEE International Conference on Neural Networks - Conference Proceedings. 2. 985 - 990 vol.2. 10.1109/IJCNN.2004.1380068. 

Contact
--------------------------------------------------------------------------------
For inquiries, collaborations, or any questions, feel free to reach out through email:

- **Email:** [alvarogrgz@gmail.com](mailto:alvarogrgz@gmail.com)
