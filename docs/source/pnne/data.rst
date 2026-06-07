:parenttoc: True

.. _pnne_data:

Data
====

|

We share four datasets as examples for users to try out the pretrained NNE. You can find them (mat files) in the “sample_data” folder at this `GitHub directory <https://github.com/pnnehome/code_matlab>`_. These datasets are used in Wei and Jiang (2025) and come from public sources. More detailed descriptions of these datasets can be found in the paper too.

|

Description of the datasets
---------------------------

Expedia - destination 1
~~~~~~~~~~~~~~~~~~~~~~~~~

This dataset comes from `a Kaggle contest <https://www.kaggle.com/competitions/expedia-personalized-sort/overview>`_ that studies hotel searches and bookings on Expedia.com. It has been used in several papers to study consumer online search behaviors. This dataset here includes the search sessions for the largest travel destination in this contest. There are :math:`n = 1258` sessions. There are 3 product attributes, 2 consumer attributes, and 1 advertising attribute.

Expedia - destination 2
~~~~~~~~~~~~~~~~~~~~~~~~~

This dataset includes the search sessions for the second largest travel destination in the same Kaggle contest as above. There are :math:`n = 897` sessions, slightly smaller than the current requirement of the pretrained NNE (see :ref:`code <pnne_code>`). Despite this small shortfall, the pretrained NNE seems to work well in this case.

Trivago - desktop channel
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This dataset comes from the `ACM RecSys Challenge <https://recsys.acm.org/recsys19/challenge/>`_ that analyzes user sessions on Trivago.com. This dataset here includes the search sessions made on the desktop channel. The setting of Trivago does not exactly fit the standard sequential search model (because a purchase is not well defined). Nevertheless, it provides a good place to try out the pretrained NNE.

Trivago - mobile channel
~~~~~~~~~~~~~~~~~~~~~~~~~~

This dataset includes the search sessions made on the mobile channel from the same RecSys Challenge above.

|

Papers
------

Wei, Yanhao ‘Max’ and Zhenling Jiang (2025), “Pretraining Estimators for Structural Models: Application to Consumer Search.” `arXiv <https://arxiv.org/abs/2505.00526>`__
