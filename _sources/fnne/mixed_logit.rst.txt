:parenttoc: True

.. _fnne_mixed_logit:

Mixed logit model
=================

|

This application uses **full-information NNE** to estimate a mixed logit model. The Matlab code is
available in this `GitHub repository <https://github.com/nnehome>`_.
In this mixed logit model, the likelihood is easy to simulate, so full-information NNE does not really
have an advantage over SMLE. But it offers a good setting to **demonstrate and understand** the method.

More details of this application can be found in our paper referenced on the :ref:`Full-information NNE <fnne>` page.

.. note::

   **Scaffold page.** This page mirrors the structure of the :ref:`NNE consumer search <code_consumer_search>`
   code documentation. Fill in the example workflow and the file-by-file descriptions below with the
   contents of the mixed-logit repository (replace the placeholder GitHub link above with the real one).

|

An example to use the code
--------------------------

Run the scripts in order. This example is a Monte Carlo experiment that uses full-information NNE to
estimate the mixed logit parameters from a simulated dataset.

.. code-block:: console

   >> monte_carlo_data        % simulate a dataset
   >> fnne_gen                % generate the training examples (simulate datasets across theta draws)
   >> fnne_train              % train the CNN and apply it to the data

|

Description of each file
------------------------

``model_mixed_logit.m``
"""""""""""""""""""""""

*Describe the function that codes the mixed logit model (inputs, outputs).*

``fnne_gen.m``
""""""""""""""

*Describe how training examples* :math:`\{\boldsymbol{\theta}^{(\ell)}, \mathcal{D}^{(\ell)}\}` *are generated.*

``fnne_train.m``
""""""""""""""""

*Describe how the two-part CNN architecture is trained and applied to the data.*
