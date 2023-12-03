:parenttoc: True

.. _code_2:

Simple AR1 model
=================

|

The concept of NNE can be illustrated with a simple AR1 model: :math:`y_{i}={\beta}y_{i-1}+\epsilon_{i}`. The model is simple enough that it won't see computational or accuracy gains from NNE. But the simplicity allows NNE to be more easily illustrated. The code is also easier-to-follow than that in :ref:`consumer search <code_1>`. You can find the Matlab (2023b) code at this `GitHub <https://github.com/nnehome/nne-matlab>`_ repository. Below we provide description of the code files.

More details of this application can be found in our paper referenced in :ref:`home <home>` page.

|

An example to use the code:
----------------------------

Run the three scripts in order as follows. This example is a Monte Carlo experiment that uses NNE to estimate the AR1 from a simulated dataset.

.. code-block:: console

    >> monte_carlo_data		% simulate a AR1 times series and save it in data.mat
    >> nne_gen			% generate the training examples for NNE and save them in nne_training.mat
    >> nne_train		% train a neural net and apply it to data.mat

Two functions are used in these scripts: ``model.m`` and ``moments.m``, which are explained in the description below.

..
	The main code scripts are ``nne_gen.m`` and ``nne_train.m``. Other files are the supporting functions used by these scripts.

|

Description of each file:
--------------------------

``model.m``
"""""""""""""""""""""""

This function codes the simple AR1 model.

.. code-block:: console

    y = model(beta)

* Input ``beta``:  the coefficient in the AR1 model.
* Output ``y``: simulated time series collected in a vector.

|

``moments.m``
""""""""""""""

This function summarizes data into a set of moments (used in Step 2 in the procedure on :ref:`home<home>` page).

.. code-block:: console

    output = moments(y)
    
* Input ``y``: times-series vector as described above for ``model.m``.

* Output: the value of the moment(s).

|

``monte_carlo_data.m``
""""""""""""""""""""""""""

This script simulates an AR1 time series under a "true" value of  :math:`\beta`, for the purpose of Monte Carlo experiments. It uses the function ``model.m`` to simulate the time series. The time series is saved in a file ``data.mat``.

|

``nne_gen.m``
""""""""""""""

This script generates the training and validation examples (Steps 1 & 2 in the procedure on :ref:`home<home>` page).

* It uses ``model.m`` to simulate the time-series data in each training or validation example.
* It uses ``moments.m`` to summarize data in each training or validation example.
* At the end, the training and validation examples are saved in a file ``nne_training.mat``.

|

``nne_train.m``
""""""""""""""""

This script trains a shallow neural net (Steps 3 & 4 in the procedure on :ref:`home<home>` page).

* It loads the training and validation examples from ``nne_training.mat`` (created by ``nne_gen.m``).
* Validation loss is reported. You can use it to choose hyperparameters, such as the number of hidden nodes.
* At the end, it applies the trained neural net on ``data.mat`` to recover the value of :math:`\beta`.

|

|

