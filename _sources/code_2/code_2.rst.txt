:parenttoc: True

.. _code_2:

Simple AR1 model
=================

|

The concept of NNE can be illustrated with a simple AR1 model: :math:`y_{i}={\beta}y_{i-1}+\epsilon_{i}`. The model is simple enough that it won't see computational or accuracy gains from NNE. But the simplicity allows NNE to be more easily illustrated. The code is easier-to-follow than that in :ref:`consumer search <code_1>`, and thus is useful for a quicker look at how NNE actually works. This `GitHub <https://github.com/nnehome/nne-matlab>`_ page provides the Matlab (2023b) code. Below we provide description of the code. 

For more details of this application, see our paper referred in :ref:`home <home>` page.

|

An example to use the code:
----------------------------

Run the three scripts in order as follows. This example is a Monte Carlo experiment that uses NNE to estimate the AR1 from a simulated dataset.

.. code-block:: console

    >> nne_gen		% generate the training examples for NNE, saved in nne_training.mat
    >> nne_train	% train a neural net, saved in nne_trained.mat
    >> nne_use		% apply the trained neural net on a simulated time series

Two functions are used in these scripts: ``model.m`` and ``moments.m``, which we explain below.

..
	The main code scripts are ``nne_gen.m``, ``nne_train.m``, and ``nne_use.m``. These three scripts correspond to steps 1 & 2, step 3, and step 4, respectively, in the step-by-step procedure of NNE listed on :ref:`home<home>` page. Other files are the supporting functions used by these scripts.

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

* Output: the values of the moments.

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

This script trains a shallow neural net (Step 3 in the procedure on :ref:`home<home>` page).

* It loads the training and validation examples from ``nne_training.mat`` (created by ``nne_gen.m``).
* Validation loss is reported. You can use it to choose hyperparameters, most notably the number of hidden nodes.
* At the end, the trained neural net is saved in a file ``nne_trained.mat``.

|

``nne_use.m``
""""""""""""""

This script applies the trained neural net (Step 4 in the procedure on :ref:`home<home>` page).

* It loads the trained neural net from ``nne_trained.mat`` (created by ``nne_train.m``).
* It generates Monte Carlo data under a "true" value of :math:`\beta`.
* Then, it applies the trained neural net on the Monte Carlo data to recover the value of :math:`\beta`.

|

