:parenttoc: True

.. _fnne_mixed_logit:

Mixed logit model
=================

|

Below is documentation for the Matlab code in "full_info_NNE_mixed_logit" folder at this `GitHub directory <https://github.com/nnehome/fnne-matlab-code>`__. The code uses full-info NNE to estimate a mixed logit model. Full-info NNE shows no computational or accuracy advantages here, but it serves as a good example to illustrate that the two-part architecture works in practice.

Workflow
---------

The following shows how to conduct a Monte Carlo experiment that estimates the mixed logit model on a simulated dataset.

.. code-block:: console

   >> monte_carlo_data        % simulate a dataset using mixed logit model and save it in data.mat
   >> nne_gen                 % generate the training examples
   >> nne_train               % train a neural net and apply it to the data


Description of each file
------------------------

``model_mixed_logit.m``
"""""""""""""""""""""""

This function codes the mixed logit model.

.. code-block:: console

    Y = mix_logit_model(rs, par, X, consumer_idx)
    
* Inputs:

  * ``rs``: a random stream (to control randomness)
  * ``par``: mixed logit model parameter vector
  * ``X``: product attributes
  * ``consumer_idx``: indices of consumers
 
* Outputs:

  * ``Y``: dummies indicating if products are bought
  
  
``monte_carlo_data.m``
"""""""""""""""""""""""

This script simulates a dataset using the mixed logit model under an assumed "true" parameter.

* It uses ``model_mixed_logit.m`` to simulate the data.


``nne_gen.m``
""""""""""""""

This script generates the training, validation, and test examples.

* It uses ``model_mixed_logit.m`` to simulate the examples.


``nne_train.m``
""""""""""""""""

This script trains a neural net, using the examples from ``nne_gen.m``.

* Validation loss is reported. You can use it to choose neural net hyperparameters, such as the numbers of hidden nodes.
* After training, it draws parameter recovery plots using the test examples.
* After training, it applies the neural net on ``data.mat``.
* In the end, it saves the trained neural net to ``trained_nne.mat``.


``learn.m``
""""""""""""""""

This function codes the training loop, and is used by ``nne_train.m``.
This is a custom training loop based on Matlab's built-in back-propagation and adam algorithms.

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
