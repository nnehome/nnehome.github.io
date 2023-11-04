:html_theme.sidebar_secondary.remove:

.. pNNE documentation master file, created by
   sphinx-quickstart on Mon Sep 18 12:08:20 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 3

.. toctree::
   :hidden:

    Home <home/home>
    Code <code/code>


Welcome to NNE
===============

|

This website describes the neural net estimator (NNE) to estimate structural models, as proposed 
in Wei and Jiang (2023). It provides a computationally-light alternative to simulated maximum likelihood 
or simulated method of moments. NNE is especially suitable for cases where many simulations are needed to 
evaluate likelihood/moment functions.

|

Overview of NNE
---------------
We write down a structural model: ``y = g(x, ϵ; θ)``. The goal of estimation is to obtain the parameter :math:`{θ}` 
after observing the covariates :math:`{x}` and outcome ``y: {y,x} → θ``.

The key idea of NNE is to use neural nets to directly learn the mapping from data to parameters. 
The graph below provides an overview of NNE.

.. math::
   :label: neural-net-training

   \begin{align*}
   \text{train a neural net } f(\cdot) \quad
   \begin{cases}
   \boldsymbol{\theta}^{(1)} \xrightarrow{\boldsymbol{g}(\boldsymbol{x}_{i},\boldsymbol{\varepsilon}_{i}^{(1)};\boldsymbol{\theta}^{(1)})} & \{\boldsymbol{y}_{i}^{(1)},\boldsymbol{x}_{i}\}_{i=1}^{n} \xrightarrow{\text{moments}} \boldsymbol{m}^{(1)} \xrightarrow{\text{neural net}} \widehat{\boldsymbol{\theta}}^{(1)} \\
   \boldsymbol{\theta}^{(2)} \xrightarrow{\hspace{6em}} & \{\boldsymbol{y}_{i}^{(2)},\boldsymbol{x}_{i}\}_{i=1}^{n} \xrightarrow{\hspace{4em}} \boldsymbol{m}^{(2)} \xrightarrow{\hspace{4.3em}} \widehat{\boldsymbol{\theta}}^{(2)} \\
   \vdots & \vdots \\
   \boldsymbol{\theta}^{(L)} \xrightarrow{\hspace{6em}} & \{\boldsymbol{y}_{i}^{(L)},\boldsymbol{x}_{i}\}_{i=1}^{n} \xrightarrow{\hspace{4em}} \boldsymbol{m}^{(L)} \xrightarrow{\hspace{4em}} \widehat{\boldsymbol{\theta}}^{(L)}
   \end{cases}
   \end{align*}

.. math::
   :label: neural-net-application

   \begin{align*}
   \text{apply } f(\cdot) \text{on real data} \quad \quad
   \begin{cases}
   \{\underbrace{\boldsymbol{y}_{i},\boldsymbol{x}_{i}}_{\text{real data}}\}_{i=1}^{n} \xrightarrow{\text{moments}} \boldsymbol{m} \xrightarrow{\text{neural net}} \underbrace{\widehat{\boldsymbol{\theta}}}_{\text{estimate}}
   \end{cases}
   \end{align*}

Notation:
 :math:`\boldsymbol{\theta}^{(\ell)}` drawn from a space :math:`\Theta`;
 :math:`\boldsymbol{y}_{i}=\boldsymbol{g}(\boldsymbol{x}_{i},\boldsymbol{\varepsilon}_{i};\boldsymbol{\theta})` a structural model;
 :math:`\boldsymbol{y}^{(\ell)}` simulated outcome;
 :math:`\boldsymbol{m}^{(\ell)}` simulated moments;
 :math:`\widehat{\boldsymbol{\theta}}` neural net prediction




1. We draw parameter values :math:`\theta^{(l)}` uniformly from a parameter space :math:`\Theta`. Using the structural model, we can generate the outcome :math:`y^{(l)}` under :math:`\theta^{(l)}`. After repeating this procedure a number of times, we get the corresponding datasets that are generated under a range of parameter values. These datasets form the basis of the training examples where we can use to learn the mapping from data to the "correct" parameter values.

2. To make training easier, we can summarize the data :math:`\{y^{(l)}, x\}`  into data moments :math:`m^{(l)}`.

3. The neural net takes the data moments as input and predicts the parameter value underlying that dataset.

4. Once the neural net is trained, we plug in the real data moments to obtain NNE estimates :math:`\hat{\theta}`.

The neural net can output "standard errors" in addition to point estimates. We establish that this neural net estimator (NNE)
converges to limited-information Bayesian posterior when the number of training datasets L is sufficiently large. 
Besides the benefit of light computational cost, NNE is also robust to redundant moments, which is beneficial for cases where 
there lacks clear guidance on moment choices from a theoretical perspective. 

|

Applying NNE
---------------

While the method is broadly applicable to many types of structural models, we use the consumer sequential search model to illustrate 
how to use NNE. The accompanying Matlab code can be found on the Code page. These codes can be used to replicate the Monte Carlo results 
from Wei and Jiang (2023).

We describe the key functions to implement NNE. 

Generate training datasets
''''''''''''''''''''''''''

``nne_gen.m``: This function implements steps (1) and (2) from above.


.. code-block:: matlab
    :class: scrollable-code-block

    %% set up

    clear; 
    seed = 1; 
    type = 4; % denotes the type of moments in the Moments() function

    tic;

    rng(seed)

    L = 10000; % number of simulations

    set_up % generate a search dataset, save in data.mat

    load('data.mat')

    % table with (normalized) search cost and reservation utility
    curve = importdata('tableData.csv'); 

    %% simulate

    input = cell(L,1);
    label = cell(L,1);

    for l = 1:L

        theta = unifrnd(lb, ub);

        [yd, yt] = gen_seq_search(pos, z, consumer_id, theta, ...
            randn(length(consumer_id),1), randn(length(unique(consumer_id)),1), curve);

        % keep non-outlier informative draws 
        [buy_rate, search_rate] = Statistics(yd, yt, pos, consumer_id, false);

        if buy_rate > 0 && buy_rate < 1 && search_rate > 0 && search_rate < 1

            input{l} = Moments(pos, z, consumer_id, yd, yt, type);
            label{l} = theta;

        end

    end

    % remove empty cells (outliers)
    input = input(~cellfun('isempty',input));
    label = label(~cellfun('isempty',label));

    input = cell2mat(input);
    label = cell2mat(label);

    %% save 

    n = size(input,1);

    input_train = input(1:floor(n*0.9),:);
    label_train = label(1:floor(n*0.9),:);

    input_test = input(floor(n*0.9)+1:n,:);
    label_test = label(floor(n*0.9)+1:n,:);

    %% generate simulated real data (for Monte Carlo)

    [yd, yt] = gen_seq_search(pos, z, consumer_id, theta_true, ...
        randn(length(consumer_id),1), randn(length(unique(consumer_id)),1), curve);

    input_sim_real = Moments(pos, z, consumer_id, yd, yt, type);
    label_sim_real = theta_true;

    %% save training set and seed
    toc;

    time_gen = toc/60;

    save('training_set.mat', 'input_train', 'label_train', 'input_test', 'label_test', ...
                            'input_sim_real', 'label_sim_real', 'label_name', ß'time_gen')

    state = rng;
    save('RNGstate.mat','state')


Several key steps include:

- Draw :math:`\theta^{(l)}` ``theta = unifrnd(lb, ub)``.
- Simulate outcome :math:`y^{(l)}` with function ``gen_seq_search()``, which takes parameter :math:`\theta^{(l)}` and error draw :math:`\epsilon^{(l)}`. This function is specific to sequential search and can be changed to other structural models.
- Summarize the data :math:`\{y^{(l)},x\}` into data moments :math:`m^{(l)}` with function ``Moments()``. It can be adapted to generate moments in other applications.
- We use 90% as training data and the rest 10% as testing data. The inputs are the moments while the labels are the corresponding correct parameters.
- For the Monte Carlo estimation, we also generate a simulated “real” data under assumed parameter ``theta_true``. The simulated “real” data moments are calculated using function ``Moments()``.


Train a neural network
''''''''''''''''''''''

``nne_train.m``: This function implements steps (3) and (4) from above.

.. code-block:: matlab
    :class: scrollable-code-block

    %% settings

    clear; 
    L = 10000; 
    num_nodes=64;
    tic;

    load('RNGstate.mat')
    rng(state)

    learn_standard_error = true;
    batch_size = 500;

    %% read data

    load('training_set.mat')

    L_train = size(input_train, 1);
    L_test  = size(input_test, 1);

    dim_in  = size(input_train, 2);
    K = size(label_train, 2);

    if learn_standard_error
        dim_out = 2*K;        

        label_train = [label_train, zeros(L_train, K)];
        label_test  = [label_test,  zeros(L_test, K)]; 

    else
        dim_out = K;
    end


    %% training

    opts = trainingOptions( 'adam', ...
                            'ExecutionEnvironment','cpu',...
                            'LearnRateSchedule','piecewise', ...
                            'LearnRateDropFactor', 0.1, ...
                            'LearnRateDropPeriod', 200, ...
                            'InitialLearnRate' , 0.01, ...
                            'GradientThreshold', 1,...
                            'MaxEpochs', 300, ...
                            'Shuffle','every-epoch',...
                            'MiniBatchSize', batch_size,...
                            'L2Regularization', 0, ...
                            'Plots','none', ...
                            'Verbose', true, ...
                            'ValidationData', {input_test, label_test}, ...
                            'ValidationFrequency', 100);
    

    layers = [
                featureInputLayer(dim_in, 'Normalization', 'rescale-symmetric')
                fullyConnectedLayer(num_nodes)
                reluLayer
                fullyConnectedLayer(dim_out)
                normalRegressionLayer('simple', ~ learn_standard_error)
             ];

    [net, info] = trainNetwork(input_train, label_train, layers, opts);

    %% summary

    err = PredSummary(input_test, label_test, label_name, net, 'figure', 0, 'table', 1);  

    %% obtain NNE estimate

    temp = predict(net, input_sim_real, 'acceleration', 'none');
    theta = temp(1:K);

    if learn_standard_error
        se = PositiveTransform(temp(K+1:2*K));
    end

    toc;
    time_train = toc/60;

    out = [theta, se, L, info.FinalValidationLoss, time_gen, time_train];

    csvwrite(sprintf('theta_L%d_nodes%d.csv', L, num_nodes), out);

Several key steps include:

- We can ask NNE to output standard errors by setting ``learn_standard_error = true;``. It will double the dimensionality of the NNE output by including both the point estimates and the standard errors.
  
- Train the neural net with function: ``[net, info] = trainNetwork(input_train, label_train, layers, opts);``

  - ``layers`` defines the network structure (e.g., number of layers and nodes)
  - ``opts`` defines the training specification (e.g., number of epochs and batch size)
  - The loss function is defined in ``normalRegressionLayer``, which depends on whether neural net needs to learn standard errors.

- For the Monte Carlo estimation, obtain estimates for the simulated “real” data with function: ``predict(net, input_sim_real, 'acceleration', 'none');`` where ``net`` denotes the trained neural network.

