# DeepChem_Predict_Toxicity_of_Molecules

## import deepchem and numpy
```
>>> import numpy as np
>>> import deepchem as dc
```
## loading the associated toxicity datasets from deepchem module called MoleculeNet
```
>>> tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()
```
## Take a look at each output: tox21_tasks
### Each of the 12 tasks here corresponds with a particular biological experiment
```
>>> tox21_tasks
['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

>>> len(tox21_tasks)
12

```
## Take a look at each output: tox21_datasets
### These datasets correspond to the training, validation, and test sets
### Now split up these datasets correctly
```
>>> tox21_datasets
(<deepchem.data.datasets.DiskDataset object at 0x7fbd699b9f50>, <deepchem.data.datasets.DiskDataset object at 0x7fbd6826f610>, <deepchem.data.datasets.DiskDataset object at 0x7fbd6826f5d0>)

>>> train_dataset, valid_dataset, test_dataset = tox21_datasets
```
### Check the shapes of X, y vectors
```
>>> train_dataset.X.shape
(6264, 1024)
>>> valid_dataset.X.shape
(783, 1024)
>>> test_dataset.X.shape
(784, 1024)
>>> np.shape(train_dataset.y)
(6264, 12)
>>> np.shape(valid_dataset.y)
(783, 12)
>>> np.shape(test_dataset.y)
(784, 12)
>>> train_dataset.w.shape
(6264, 12)
>>> np.count_nonzero(train_dataset.w)
62166
>>> np.count_nonzero(train_dataset.w == 0)
13002
```
## Take a look at each output: transformers
### BalancingTransformer adjusts the weights for individual data points so that the total weight assigned to every class is the same. That way, the loss function has no systematic preference for any one class. The loss can only be decreased by learning to correctly distinguish between classes.
```
>>> transformers
[<deepchem.trans.transformers.BalancingTransformer object at 0x7fbd6620e590>]
```

## construct a MultitaskClassi fier in DeepChem
### the Tox21 dataset has 12 tasks and 1,024 features for each sample. layer_sizes is a list that sets the number of fully connected hidden layers in the net‐ work, and the width of each one. In this case, we specify that there is a single hidden layer of width 1,000
```
>>> model = dc.models.MultitaskClassifier(n_tasks=12, n_features=1024,layer_sizes=[1000])
```
## Fitting our MultitaskClassifier object
### nb_epoch=10 says that 10 epochs of gradient descent training will be conducted. An epoch refers to one complete pass through all the samples in a dataset
```
>>> model.fit(train_dataset, nb_epoch=10)
>>> 0.32098957101504005
```

## Evaluate the performance of the trained model
### our score on the training set (0.96) is much better than our score on the test set (0.79). This shows the model has been overfit.
```
>>> metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

>>> train_scores = model.evaluate(train_dataset, [metric], transformers)
computed_metrics: [0.9952200870444559, 0.9982939948619138, 0.9738771032181113, 0.9913682552848295, 0.9307416396082908, 0.990545025549183, 0.9960264579749474, 0.9364993765325166, 0.9930224000055679, 0.983917908505803, 0.964842045029749, 0.9849303574425239]

>>> test_scores = model.evaluate(test_dataset, [metric], transformers)
computed_metrics: [0.7633277992990224, 0.8258163322654977, 0.8740590681309309, 0.7947456562696253, 0.7085775289221987, 0.7790790790790791, 0.6639357828533599, 0.6700437252073567, 0.8549690917736568, 0.6990359261700725, 0.8563975723312272, 0.7309220251293422]

>>> print(train_scores)
{'mean-roc_auc_score': 0.9782737209214911}

>>> print(test_scores)
{'mean-roc_auc_score': 0.7684091322859473}
```



