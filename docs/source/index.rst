.. _family-home:

Welcome to home of NNE
======================

This website provides the code and some documentation for the neural net estimator (NNE), a general
approach that uses neural nets to estimate structural econometric models.

Please select the type of NNE below.

.. raw:: html

   <div class="method-cards">

     <div class="method-card accent-nne">
       <h3>Original NNE (limited-information)</h3>
       <p>Based on <a class="inline-link" href="https://pubsonline.informs.org/doi/10.1287/mksc.2022.0360"
             target="_blank" rel="noopener">&ldquo;Estimating Parameters of Structural Models Using Neural Nets,&rdquo;
          Wei and Jiang (2025), Marketing Science, 44(1).</a>
          This NNE uses researcher-specified moments as input to the neural net. It is most useful
          when the researcher has clear intuition about what data moments identify the structural
          model.</p>
       <span class="method-open">Open &rarr;</span>
       <a class="card-stretch" href="nne/index.html" aria-label="Open Original NNE"></a>
     </div>

     <div class="method-card accent-pnne">
       <h3>Pre-trained NNE</h3>
       <p>Based on <a class="inline-link" href="https://arxiv.org/abs/2505.00526"
          target="_blank" rel="noopener">&ldquo;Pre-Training Estimators for Structural Models: Application to Consumer
          Search,&rdquo; Wei and Jiang (2025)</a>. This NNE pretrains a neural net
          for a given structural model, so users can estimate the structural model
          at negligible cost &mdash; as easy as running a reduced-form regression.</p>
       <span class="method-open">Open &rarr;</span>
       <a class="card-stretch" href="pnne/index.html" aria-label="Open Pre-trained NNE"></a>
     </div>

     <div class="method-card accent-fnne">
       <h3>Full-information NNE</h3>
       <p>Based on <a class="inline-link"
          href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6774178"
          target="_blank" rel="noopener">&ldquo;Estimating and Assessing Identification of Structural Models via Deep
          Learning,&rdquo; Wei and Jiang (2026)</a>. This NNE uses an entire dataset
          as input, and automatically exploits variation in data. It is useful for not only estimating
          but also assessing the identification of a structural model.</p>
       <span class="method-open">Open &rarr;</span>
       <a class="card-stretch" href="fnne/index.html" aria-label="Open Full-information NNE"></a>
     </div>

   </div>

|

Contact
-------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - `Yanhao 'Max' Wei <https://www.yanhaowei.com/>`__
     - Marshall School of Business, University of Southern California.
   * - `Zhenling Jiang <https://jiangzhenling.com/>`__
     - The Wharton School, University of Pennsylvania.

|

.. toctree::
   :hidden:

   NNE <nne/index>
   Pre-trained NNE <pnne/index>
   Full-information NNE <fnne/index>
