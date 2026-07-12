:parenttoc: True

.. _pnne_data:

Data
====

|

We share four datasets as examples for users to try out the pretrained NNE. You can find them (Matlab files) in the 'sample_data' folder at this `GitHub directory <https://github.com/pnnehome/code_matlab>`_. These datasets all come from public sources. More detailed descriptions of these datasets can be found in the paper.


Description of the datasets
---------------------------

Expedia - destination 1
~~~~~~~~~~~~~~~~~~~~~~~~~

This dataset comes from `a Kaggle contest <https://www.kaggle.com/competitions/expedia-personalized-sort/overview>`_ based on Expedia.com data (which have been used by several papers to study consumer online search behaviors). This dataset here focuses on the search sessions for the largest travel destination in this contest. There are :math:`n = 1258` sessions, 3 product attributes, 2 consumer attributes, and 1 advertising attribute.

Expedia - destination 2
~~~~~~~~~~~~~~~~~~~~~~~~~

This dataset includes the search sessions for the 2nd largest travel destination in the same Kaggle contest as above. There are :math:`n = 897` sessions, slightly below the current requirement of :math:`n` by the pretrained NNE (see :ref:`here <pnne>`). Despite this, the pretrained NNE seems to work well.

Trivago - desktop channel
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This dataset comes from the `ACM RecSys Challenge <https://recsys.acm.org/recsys19/challenge/>`_ based on the user data from Trivago.com. This dataset here includes the search sessions made on the desktop channel.

Trivago - mobile channel
~~~~~~~~~~~~~~~~~~~~~~~~~~

This dataset includes the search sessions made on the mobile channel from the same RecSys Challenge above.

|

