:parenttoc: True

.. _fnne_mixed_logit:

Mixed logit model
=================

|

Below is documentation for the Matlab code in "mixed_logit" folder at this `GitHub directory <https://github.com/nnehome/fnne-matlab-code>`__. The code uses full-info NNE to estimate a mixed logit model. Because the likelihood for mixed logit is relatively easy to simulate, full-info NNE shows no advantages in accuracy or computation here. But this setting is a good example to illustrate how full-info NNE works in practice.

Workflow
---------

The following commands run a Monte Carlo experiment that estimates the mixed logit model on a simulated dataset.

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

This script creates a dataset for Monte Carlo experiment. It uses ``model_mixed_logit.m`` to simulate the dataset under a "true" parameter vector, and then saves the dataset in ``data.mat``.


``nne_gen.m``
""""""""""""""

This script generates the training, validation, and test examples. It uses ``model_mixed_logit.m`` to simulate the examples.


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
