:parenttoc: True

.. _pnne_data:

Data
====

|

To use the pre-trained NNE, organize your data into five arrays. With :math:`n` consumers and
:math:`J` options each (row order: product :math:`j` nested within consumer :math:`i`):

.. list-table::
   :widths: 22 18 60
   :header-rows: 1
   :class: table-header-centered

   * - Array
     - Size
     - Description
   * - ``consumer_idx``
     - :math:`(n \times J) \times 1`
     - Maps each row to its consumer :math:`i`.
   * - ``Y``
     - :math:`(n \times J) \times 2`
     - ``[searched, bought]`` indicators for each product.
   * - ``Xp``
     - :math:`(n \times J) \times K_p`
     - Product attributes.
   * - ``Xa``
     - :math:`(n \times J) \times K_a`
     - Advertising attributes.
   * - ``Xc``
     - :math:`n \times K_c`
     - Consumer attributes.

|

Sample datasets
---------------

The ``sample_data`` folder in the `GitHub repository <https://github.com/nnehome>`_ contains
real datasets used in Wei and Jiang (2025), all derived from public sources:

.. list-table::
   :widths: 28 12 60
   :header-rows: 1
   :class: table-header-centered

   * - Dataset
     - Sessions
     - Notes
   * - **Expedia — Destination 1**
     - 1,258
     - Hotel search sessions (Kaggle competition); 3 product, 2 consumer, 1 advertising attribute.
   * - **Expedia — Destination 2**
     - 897
     - Second-largest destination in the same contest; below the typical :math:`n \geq 1{,}000` guideline but reportedly functional.
   * - **Trivago — Desktop**
     - —
     - ACM RecSys Challenge, desktop interface; purchase definition is not a perfect fit for the standard search model.
   * - **Trivago — Mobile**
     - —
     - Mobile channel from the same RecSys Challenge.

.. note::

   **Migrated section.** Copy the exact array-construction examples and the per-dataset details from
   the original ``pnnehome`` documentation / repository as needed.
