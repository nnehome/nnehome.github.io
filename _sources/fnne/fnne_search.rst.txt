:parenttoc: True

.. _fnne_search:

Search with unobserved heterogeneity
=====================================

|

Below is documentation for the Matlab code in "full_info_NNE_search" folder at this `GitHub repository <https://example.com>`__. The code uses full-info NNE to estimate a search model with unobserved consumer heterogeneity.
|

Workflow
---------

The following shows how to estimate the search model on a real dataset (in ``data.mat``).

.. code-block:: console

   >> nne_gen                 % generate the training examples
   >> nne_train               % train the neural net and apply it to the data

|

Description of each file
------------------------

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
  * ``consumer_id``: indices of consumers (or search sessions)
 
* Outputs:

  * ``Y``: dummies indicating searches, purchases, first-search, and last-search
  * ``stat``: summary statistics

|

``nne_gen.m``
""""""""""""""

This script generates the training, validation, and test examples.

* It uses ``search_ht_model.m`` to simulate the data.
* It uses Matlab's built-in bit-to-integer encoding on ``Y`` to save memory, decoded later during training.

|

``nne_train.m``
""""""""""""""""

This script trains a neural net, using the examples from ``nne_gen.m``.

* Validation loss is reported. You can use it to choose neural net hyperparameters, such as the numbers of hidden nodes.
* After training, it draws parameter recovery plots using the test examples.
* After training, it applies the neural net on ``data.mat``.
* In the end, it saves the trained neural net to ``trained_nne.mat``.

|

``learn.m``
""""""""""""""""

This function codes the training loop, and is used by ``nne_train.m``.
This is a custom training loop based on Matlab's built-in back-propogation and adam algorithms.

.. code-block:: console

    [ema_net, train_pred, val_pred, test_pred] = ...
                learn(net, opt, nne, train_dataY, train_label, val_dataY, val_label, test_dataY)
                
* Inputs:

  * ``net``: the initial neural net
  * ``opt``: options including batch size, number of iterations, etc.
  * ``train_dataY``, ``train_label``: training examples
  * ``val_dataY``, ``val_label``: validation examples
  * ``test_dataY``:  test examples
 
* Outputs:

  * ``ema_net``: the final trained neural net
  * ``train_pred``: predictions for training examples
  * ``val_pred``: predictions for validation examples
  * ``test_pred``: predictions for test examples
  
|
