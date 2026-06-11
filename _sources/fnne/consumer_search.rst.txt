:parenttoc: True

.. _fnne_consumer_search:

Consumer search
===============

|

This application uses **full-information NNE** to estimate a sequential search model with
**unobserved consumer heterogeneity**. The Matlab code is available in this
`GitHub repository <https://github.com/nnehome>`_.

More details of this application can be found in our paper referenced on the :ref:`Full-information NNE <fnne>` page.

.. note::

   **Scaffold page.** This page mirrors the structure of the :ref:`NNE consumer search <code_consumer_search>`
   code documentation. Fill in the example workflow and the file-by-file descriptions below with the
   contents of the search-model repository (replace the placeholder GitHub link above with the real one).

|

An example to use the code
--------------------------

Run the scripts in order. This example is a Monte Carlo experiment that uses full-information NNE to
estimate the search model parameters from a simulated dataset.

.. code-block:: console

   >> monte_carlo_data        % simulate a dataset
   >> fnne_gen                % generate the training examples (simulate datasets across theta draws)
   >> fnne_train              % train the CNN and apply it to the data

|

Description of each file
------------------------

``model_seq_search.m``
""""""""""""""""""""""

*Describe the sequential search model with unobserved consumer heterogeneity (inputs, outputs).*

``fnne_gen.m``
""""""""""""""

*Describe how training examples* :math:`\{\boldsymbol{\theta}^{(\ell)}, \mathcal{D}^{(\ell)}\}` *are generated.*

``fnne_train.m``
""""""""""""""""

*Describe how the two-part CNN architecture is trained and applied to the data.*
