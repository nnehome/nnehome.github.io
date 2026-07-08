.. _family-home:

Welcome to Neural Network Estimator (NNE) 
==========================================

This website provides the code and documentation for the neural net estimator (NNE), a general
approach that uses neural nets to estimate structural econometric models.

Please select the type of NNE below.

.. raw:: html

   <div class="method-cards">
   
     <div class="method-card accent-fnne">
       <h3>Full-information NNE</h3>
       <p>Based on <span class="note-text">"Estimating and Assessing Identification of Structural Models via Deep
          Learning."</span> This NNE uses an entire dataset
          as input, and automatically exploits variation in data. It is useful for not only estimating but also assessing 
          the identification of a structural model.</p>
       <span class="method-open">Open &rarr;</span>
       <a class="card-stretch" href="fnne/index.html" aria-label="Open Full-information NNE"></a>
     </div>
     
     <div class="method-card accent-pnne">
       <h3>Pretrained NNE</h3>
       <p>Based on <span class="note-text">"Pre-Training Estimators for Structural Models: Application to Consumer
          Search."</span> This NNE pretrains a neural net
          for a given structural model, so users can estimate the structural model
          at negligible cost &mdash; as easy as running a reduced-form regression.</p>
       <span class="method-open">Open &rarr;</span>
       <a class="card-stretch" href="pnne/index.html" aria-label="Open Pre-trained NNE"></a>
     </div>

     <div class="method-card accent-nne">
       <h3>Original NNE (limited-information)</h3>
       <p>Based on <span class="note-text">"Estimating Parameters of Structural Models Using Neural Nets."</span>
          This NNE uses researcher-specified moments as input to the neural net.</p>
       <span class="method-open">Open &rarr;</span>
       <a class="card-stretch" href="nne/index.html" aria-label="Open Original NNE"></a>
     </div>

   </div>

|

Contact
---------------

To be added ...

.. toctree::
   :hidden:

   Full-information NNE <fnne/index>
   Pre-trained NNE <pnne/index>
   NNE <nne/index>
   
