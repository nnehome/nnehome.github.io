:parenttoc: True

.. _fnne_search:

Search with unobserved heterogeneity
=====================================

|

Below is documentation for the Matlab code in "search_het" folder at this `GitHub directory <https://github.com/nnehome/fnne-matlab-code>`__. The code uses full-info NNE to estimate a search model with unobserved consumer heterogeneity.

Workflow
---------

The following commands estimate the search model on a synthetic dataset (in ``data.mat``).

.. code-block:: console

   >> nne_gen                 % generate the training examples
   >> nne_train               % train the neural net and apply it to the data


Description of each file
------------------------


``data.mat``
""""""""""""""""""""""

This file is a simulated dataset that resembles the MSOM Research Challenge dataset used in the paper.


``search_ht_model.m``
""""""""""""""""""""""

This function codes the search model with unobserved consumer heterogeneity.

.. code-block:: console

    [Y, stat] = search_ht_model(rs, par, curve, X, consumer_idx)
    
* Inputs:

  * ``rs``: a random stream (to control randomness)
  * ``par``: the search model parameter vector
  * ``curve``: lookup table for reservation utility, available from ``curve.mat``
  * ``X``: product attributes
  * ``consumer_idx``: indices of consumers (or search sessions)
 
* Outputs:

  * ``Y``: dummies indicating searches, purchases, first-searches, and last-searches
  * ``stat``: summary statistics


``nne_gen.m``
""""""""""""""

This script generates the training, validation, and test examples.

* It uses ``search_ht_model.m`` to simulate the data.
* It uses Matlab's built-in bit2int encoding on ``Y`` to save memory.


``nne_train.m``
""""""""""""""""

This script trains a neural net, using the examples from ``nne_gen.m``.

* Validation loss is reported. We can use this loss to choose neural net hyperparameters (e.g., numbers of hidden nodes).
* It draws the parameter recovery plots using the test examples.
* It applies the trained neural net on ``data.mat``.
* It saves the trained neural net to ``trained_nne.mat``.


``learn.m``
""""""""""""""""

This function codes the training loop, and is used by ``nne_train.m``.
This is a custom training loop based on Matlab's built-in back-propagation and adam algorithms.

.. code-block:: console

    [ema_net, train_pred, val_pred, test_pred] = ...
                learn(net, opt, nne, train_dataY, train_label, val_dataY, val_label, test_dataY)
                
* Inputs:

  * ``net``: the initial neural net
  * ``opt``: training options such as batch size, number of iterations, etc.
  * ``nne``: a structure storing some settings of NNE, created by ``nne_gen.m``.
  * ``train_dataY``, ``train_label``: training examples
  * ``val_dataY``, ``val_label``: validation examples
  * ``test_dataY``:  test examples
 
* Outputs:

  * ``ema_net``: the final trained neural net
  * ``train_pred``: predictions for training examples
  * ``val_pred``: predictions for validation examples
  * ``test_pred``: predictions for test examples
  
|
