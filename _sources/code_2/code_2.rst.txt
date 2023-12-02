:parenttoc: True

.. _code_2:

Simple AR1 model
=================

The concept of NNE can be illustrated in estimating a simple AR1 model: :math:`y_{i}={\beta}y_{i-1}+\epsilon_{i}`. The model is simple enough that it won't see computational or accuracy gains from NNE. But the simplicity allows NNE to be more easily illustrated. This `GitHub <https://github.com/nnehome/nne-matlab>`_ page provides the Matlab (2023b) code. The code is easier-to-follow than that in :ref:`consumer search <code_1>`, and is useful for a quicker look at how NNE works in an application. See `our paper <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3496098#>`_ for more details of the application.

The :ref:`home<home>` page has listed 4 steps to apply NNE. The corresponding files are ``nne_gen.m`` (Step 1 & 2), ``nne_train.m`` (Step 3), and ``nne_use.m`` (Step 4). Other files are supporting functions used by these scripts.

Description of code files
--------------------------

``model.m``
"""""""""""""""""""""""

This function codes the simple AR1 model.

.. code-block:: console

    y = model(beta)

* Input ``beta``:  the coefficient in the AR1 model.
* Output ``y``: simulated time series in a vector.

``moments.m``
""""""""""""""

This function summarizes data into a set of moments (used in Step 2 in the procedure on :ref:`home<home>` page).

.. code-block:: console

    output = moments(y)
    
* Input ``y``: times-series vector as described above for ``model.m``.

* Output: the values of the moments.

``nne_gen.m``
""""""""""""""

This script generates the training and validation examples (Step 1 & 2 in the procedure on :ref:`home<home>` page).

* It uses ``model.m`` to simulate the time-series data in each training or validation example.
* It uses ``moments.m`` to summarize data in each training or validation example.
* At the end, the training and validation examples are saved in a file ``nne_training.mat``.

``nne_train.m``
""""""""""""""""

This script trains a shallow neural net (Step 3 in the procedure on :ref:`home<home>` page).

* It loads the training and validation examples from ``nne_training.mat`` (created by ``nne_gen.m``).
* Validation loss is reported. You can use it to choose hyperparameters, most notably the number of hidden nodes.
* At the end, the trained neural net is saved in a file ``nne_trained.mat``.

``nne_use.m``
""""""""""""""

This script applies the trained neural net (Step 4 in the procedure on :ref:`home<home>` page).

* It loads the trained neural net from ``nne_trained.mat`` (created by ``nne_train.m``).
* It generates Monte Carlo data under a "true" value of :math:`\beta`.
* Then, it applies the trained neural net on the Monte Carlo data to recover the value of :math:`\beta`.

|

