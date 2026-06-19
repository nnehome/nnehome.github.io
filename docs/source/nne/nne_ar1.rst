:parenttoc: True

.. _nne_ar1:

Simple AR1 model
=================

|

Below is documentation for the Matlab code in "NNE_AR1" folder at this `GitHub repository <https://github.com/nnehome/nne-matlab-code>`_. The code uses NNE to estimate a simple AR1 model. The AR1 model is :math:`y_{i}={\beta}y_{i-1}+\epsilon_{i}`. This is a toy example where we don't see computational or accuracy gains from NNE. But the simplicity allows NNE to be more easily understood.

|

Workflow
----------

The following shows how to conduct a Monte Carlo experiment that estimates the AR1 on a simulated dataset.

.. code-block:: console

    >> monte_carlo_data		% simulate an AR1 times series and save it in data.mat
    >> nne_gen			% generate the training examples for NNE and save them in nne_training.mat
    >> nne_train		% train a neural net and apply it to data.mat

|

Description of each file
--------------------------

``model.m``
"""""""""""""""""""""""

This function codes the simple AR1 model.

.. code-block:: console

    y = model(beta)

* Input ``beta``:  the coefficient in the AR1 model.
* Output ``y``: simulated time series collected in a vector.

..  collapse:: Click to show code <collapse_header>model.m</collapse_header>

    .. code-block:: matlab
        :class: scrollable-code-block

        function y = model(beta)

        %{

        This function codes a simple AR1 model: y(i) = beta*y(i-1) + epsilon(i),
        with epsilon ~ N(0,1) and  y(1) drawn from the stationary distribution.

        Input:
            beta .. the coefficient.
        Output:
            y .. simulated time series stored in a vector of length n.

        %}

        n = 100; % number of observations (or periods).

        epsilon = randn(n,1); % error terms

        y = nan(n,1);

        y(1) = epsilon(1)/sqrt(1 - beta^2); % draw initial value

        % draw rest of the values
        for i = 2:n
            y(i) = beta*y(i-1) + epsilon(i);
        end

|

``moments.m``
""""""""""""""

This function summarizes data into a set of moment(s).

.. code-block:: console

    output = moments(y)
    
* Input ``y``: times-series vector as outputted from ``model.m``.

* Output: the value of the moment(s).

..  collapse:: Click to show code <collapse_header>moments.m</collapse_header>

    .. code-block:: matlab
        :class: scrollable-code-block

        function output = moments(y)

        %{

        This function summarizes data into moments.

        Currently the output is a single moment. To use more moments:
        (a) Change k to larger than 1 to include more lags;
        (b) Uncomment m{2}, m{3}, m{4} to include higher-order moments.

        %}

        % how many lags to use.
        k = 1;

        % lagged values of y
        x = lagmatrix(y, 1:k);
        x( isnan(x)) = 0;

        % compute moments.
        m{1} = mean(y.*x);
        % m{2} = mean(y.^2);
        % m{3} = mean(y.^2.*x);
        % m{4} = mean(y.*x.^2);

        % final output.
        output = cell2mat(m);

|

``monte_carlo_data.m``
""""""""""""""""""""""""""

This script simulates an AR1 time series under a "true" value of  :math:`\beta`, for the purpose of Monte Carlo experiments. It uses ``model.m`` and saves the time series into ``data.mat``.

..  collapse:: Click to show code <collapse_header>monte_carlo_data.m</collapse_header>

    .. code-block:: matlab
        :class: scrollable-code-block

        %{

        This script generates a Monte Carlo data for estimation of AR1 model. The
        data will be saved in data.mat.

        %}

        clear

        % set true parameter value
        beta_true = 0.6;

        % simulate the data
        y = model(beta_true);

        % save data
        save('data.mat', 'y')

|

``nne_gen.m``
""""""""""""""

This script generates the training and validation examples.

* It uses ``model.m`` to simulate the time-series data in each training or validation example.
* It uses ``moments.m`` to summarize data in each training or validation example.
* The training and validation examples are saved to ``nne_training.mat``.

..  collapse:: Click to show code <collapse_header>nne_gen.m</collapse_header>

    .. code-block:: matlab
        :class: scrollable-code-block

        %{

        This script generates the training and validation examples to be used to
        train NNE. The examples will be saved in nne_trainning.mat.

        Change 'for' to 'parfor' if parallel computing toolbox is available.

        For the illustration of NNE on AR1 model.

        %}

        clear

        %% settings

        label_name = '\beta'; % name of the AR1 parameter to be estimated
        lb = 0; % lower bound of the AR1 parameter
        ub = 0.9; % upper bound of the AR1 parameter

        %% simulate

        L = 1000; % number of training & validation examples

        % pre-allocation for training & validation examples
        input = cell(L,1);
        label = cell(L,1);

        for l = 1:L
            
            % draw the value for the AR1 parameter
            beta = unifrnd(lb, ub);
            
            % simulate the AR1 time series data
            y = model(beta);
            
            % compute moment(s) and store the result.
            input{l} = moments(y);
            label{l} = beta;
            
        end

        input = cell2mat(input);
        label = cell2mat(label);

        %% training-validation split

        L_train = floor(L*0.8); % number of training examples (80-20 split)

        input_train = input(1:L_train,:);
        label_train = label(1:L_train,:);

        input_val = input(L_train+1:L,:);
        label_val = label(L_train+1:L,:);

        %% save 

        save('nne_training.mat','input_train','label_train','input_val','label_val','label_name')

|

``nne_train.m``
""""""""""""""""

This script trains a shallow neural net.

* It loads the training and validation examples from ``nne_training.mat`` (saved by ``nne_gen.m``).
* Validation loss is reported. You can use it to choose neural net hyperparameters, such as the number of hidden nodes.
* After training, it applies the trained neural net on ``data.mat``.

..  collapse:: Click to show code <collapse_header>nne_train.m</collapse_header>

    .. code-block:: matlab
        :class: scrollable-code-block

        %{

        This script trains the neural net in NNE, and then applies the trained
        neural net on data.mat to obtain a parameter estimate.

        For the illustration of NNE on AR1 model.

        %}

        clear

        %% settings

        num_nodes = 32; % number of hidden nodes (in shallow neural net)

        %% load training & validation examples

        load('nne_training.mat')

        L_train = size(input_train, 1); % number of training examples
        L_val   = size(input_val,   1); % number of validation examples

        dim_input = size(input_train, 2); % number of inputs by neural net

        %% train a neural net

        opts = trainingOptions( 'adam', ...
                                'L2Regularization', 0, ...
                                'ExecutionEnvironment', 'cpu', ...
                                'MaxEpochs', 500, ...
                                'InitialLearnRate', 0.01, ...
                                'GradientThreshold', 1, ...
                                'MiniBatchSize', 500, ...
                                'Plots','none', ...
                                'Verbose', true, ...
                                'VerboseFrequency', 100, ...
                                'ValidationData', {input_val, label_val},...
                                'ValidationFrequency', 100);

        layers = [  featureInputLayer(dim_input)
                    fullyConnectedLayer(num_nodes)
                    reluLayer
                    fullyConnectedLayer(1)
                    regressionLayer
                    ];

        [net, info] = trainNetwork(input_train, label_train, layers, opts);

        disp("Final validation loss is: " + info.FinalValidationLoss)

        %% display figure: estimate vs. truth in validation

        pred_val = predict(net, input_val, exec='cpu');

        figure('position', [750,500,250,250])
        sgtitle('Estimate vs. Truth in Validation')
        scatter(label_val, pred_val, '.')
        xlabel(label_name)
        axis equal

        %% apply the trained neural net on data.mat

        load('data.mat') % load the data

        input = moments(y); % calculate data moments to be used as neural net input
        estimate = predict(net, input, exec='cpu'); % apply the trained neural net

        % display estimates
        result = table(estimate, 'row', {label_name}, 'var', {'Estimate'});
        disp(result)

|

|

