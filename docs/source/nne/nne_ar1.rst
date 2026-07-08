:parenttoc: True

.. _nne_ar1:

Simple AR1 model
=================

|

Below is documentation for the Matlab code in "NNE_AR1" folder at this `GitHub repository <https://github.com/nnehome/nne-matlab-code>`_. The code uses NNE to estimate a simple AR1 model. The AR1 model is :math:`y_{i}={\beta}y_{i-1}+\epsilon_{i}`. This is a toy example where we don't see computational or accuracy gains from NNE. But the simplicity allows NNE to be more easily understood.


Workflow
----------

The following shows how to conduct a Monte Carlo experiment that estimates the AR1 on a simulated dataset.

.. code-block:: console

    >> monte_carlo_data		% simulate an AR1 times series and save it in data.mat
    >> nne_gen			% generate the training examples for NNE and save them in nne_training.mat
    >> nne_train		% train a neural net and apply it to data.mat


Description of each file
--------------------------

``model.m``
"""""""""""""""""""""""

This function codes the simple AR1 model.

.. code-block:: console

    y = model(beta)

* Input ``beta``:  the coefficient in the AR1 model.
* Output ``y``: simulated time series collected in a vector.


``moments.m``
""""""""""""""

This function summarizes data into a set of moment(s).

.. code-block:: console

    output = moments(y)
    
* Input ``y``: times-series vector as outputted from ``model.m``.

* Output: the value of the moment(s).


``monte_carlo_data.m``
""""""""""""""""""""""""""

This script simulates an AR1 time series under a "true" value of  :math:`\beta`, for the purpose of Monte Carlo experiments. It uses ``model.m`` and saves the time series into ``data.mat``.


``nne_gen.m``
""""""""""""""

This script generates the training and validation examples.

* It uses ``model.m`` to simulate the time-series data in each training or validation example.
* It uses ``moments.m`` to summarize data in each training or validation example.
* The training and validation examples are saved to ``nne_training.mat``.


``nne_train.m``
""""""""""""""""

This script trains a shallow neural net.

* It loads the training and validation examples from ``nne_training.mat`` (saved by ``nne_gen.m``).
* Validation loss is reported. You can use it to choose neural net hyperparameters, such as the number of hidden nodes.
* After training, it applies the trained neural net on ``data.mat``.

|

