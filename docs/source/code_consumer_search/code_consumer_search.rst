:parenttoc: True

.. _code_consumer_search:

Consumer search
============================

|

One example application of NNE is to estimate consumer search model. This `GitHub <https://github.com/nnehome/nne-matlab-code>`_ repository provides the Matlab (2023b) code. Below we provide description of the code files. The code itself is commented as well. You are welcome to modify the code to estimate your own structural model. The GitHub repository also provides the code that uses SMLE to estimate the search model.

More details of this application (e.g., specification of the search model) can be found in our paper referenced in :ref:`home <home>` page.

|

An example to use the code:
----------------------------

Run the three scripts in order as follows. This example is a Monte Carlo experiment that uses NNE to estimate the search model parameter from a simulated dataset.

.. code-block:: console

    >> monte_carlo_data		% simulate a dataset and save it in data.mat
    >> nne_gen			% generate the training examples for NNE and save them in nne_training.mat
    >> nne_train		% train a neural net and then apply it on data.mat

Three functions are used in these scripts: ``model_seq_search.m``, ``moments.m``, and ``normalRegressionLayer.m``, which are explained in the description below.
    
..
	The main code scripts are ``nne_gen.m`` and ``nne_train.m``. The data for estimation is stored in ``data.mat``. You can use script ``monte_carlo_data.m`` to simulate data for Monte Carlo experiments. Other files are the supporting functions used by these scripts.

|

Description of each file:
--------------------------

``model_seq_search.m``
""""""""""""""""""""""""""

This function codes a sequential search model.

.. code-block:: console

    [yd, yt, order] = model_seq_search(pos, z, consumer_id, theta, curve)

* Inputs:

  * ``pos``: product ranking positions
  * ``z``: other product attributes (e.g., review rating, price)
  * ``consumer_id``: indices of consumers (or search sessions)
  * ``theta``: vector of search model parameter
  * ``curve``: relation between search cost and reservation utility, stored in ``curve_seq_search.csv``
 
* Outputs:

  * ``yd``: dummies indicating if products are searched
  * ``yt``: dummies indicating if products are bought
  * ``order``: order of search

..  collapse:: Click to show code <collapse_header>model_seq_search.m</collapse_header>

    .. code-block:: matlab
        :class: scrollable-code-block

        function [yd, yt, order] = model_seq_search(pos, z, consumer_id, theta, curve)

        %{

        This function codes the sequential search model and is largely based on the
        code used in Ursu (2018): "The Power of Rankings ..."

        Outputs: 
            yd .. search dummies
            yt .. purchase dummies
            order .. search order
        Inputs: 
            pos .. prodoct ranking positions
            z .. other product attributes
            consumer_id .. consumer indices
            theta .. model parameter
            curve .. relation between search cost and reservation utility

        %}

        %% random terms

        rows = numel(consumer_id); % number of observations
        N = numel(unique(consumer_id)); % number of consumers

        eps = randn(rows, 1); % utility shocks for inside goods
        eps0 = randn(N, 1);  % utility shocks for outside goods

        %% setup

        % number of options for each consumer
        Ji = accumarray(consumer_id,1);
        Ji = Ji(consumer_id);

        bepos = theta(end); % last par, position
        constc = theta(end - 1); % par (2nd to last): constant in search cost

        v0 = theta(end - 2); % outside option

        [~, index] = ismember(1:N, consumer_id); % index of consumers in the data

        %form utility from data and param
        eutility = z * theta(1:size(z,2))';

        utility = eutility + eps;

        % utility of the outside option
        u0 = v0 + eps0;

        % search cost c, and therefore m, only changes with pos

        pos_unique = sort(unique(pos));
        m_pos = zeros(length(pos_unique),1);
        for i = 1:length(pos_unique)
            c_i = exp(constc + log(i).*bepos);
            if c_i<curve(1,2) && c_i>=curve(end,2)
                for n = 2:length(curve)
                    if (curve(n,2) == c_i)
                        m_pos(i) = curve(n,1);
                    elseif ((curve(n-1,2)>c_i)&& (c_i>curve(n,2)))
                        m_pos(i) = (curve(n,1)+curve(n-1,1))/2;
                    end
                end
            elseif c_i>=curve(1,2)
                m_pos(i) = -c_i;
            elseif c_i<curve(end,2)
                m_pos(i) = 4.001;
            end
        end
        m = m_pos(pos);

        %reservation utilities
        r = m + eutility;

        %order by r for each consumer
        da = [consumer_id, pos, Ji, z, utility, eutility, r];
        whatr = size(da,2);
        whateu = whatr - 1;
        whatu = whateu - 1;

        order = zeros(rows,1);

        for m = 1:N
            n = index(m);
            J = Ji(n);
        %     for j = n:n+J-1
            [~, order(n:n+J-1)] = sort(da(n:n+J-1, whatr),'descend');
        %     end
        end

        o = ones(rows, 1);
        for m = 1:N
            n = index(m);
            J = Ji(n);
            for j = n:n+J-1
                o(j) = order(j) + n - 1;
            end
        end

        data = da(o, :);

        % click decisions
        yd = zeros(rows, 1);
        ydn = zeros(rows, 1);

        % order of clicks
        order = zeros(rows,1);

        % free first click
        yd(index) = 1;

        %for next click decisions: if r is higher than outside
        %option and higher than all utilities so far, then increase click d by one
        % It is ok to do this because we ordered the r's first, so we know that rn>rn+1
        for i = 1:N
            J = Ji(index(i));
            for j = 1:(J-1)
                ma = max(data(index(i):(index(i)+j-1), whatu), u0(i));
                if data(index(i)+j, whatr) > ma
                    yd(index(i)+j) = 1;
                else
                    break
                end
            end
            ydn(index(i):(index(i)+J-1)) = sum(yd(index(i):(index(i)+J-1)));
        end

        %tran decisions: if out of those clicked (the set of indices from first to
        %max) u=max, then put a 1, otherwise put zero; finally reshape
        yt = zeros(rows, 1);
        mi = zeros(rows, 1);

        for i = 1:N
            J = Ji(index(i));
            ydn_i = ydn(index(i));
        %     if ydn_i>0
                order(index(i):index(i)+ydn_i-1) = 1:ydn_i;
        %     end
            mi(index(i):index(i)+J-1) = max([data(index(i):index(i)+ydn_i-1, whatu); u0(i)]);
        end
        yt(data(:, whatu) == mi) = 1;

        [~, i] = ismember((1:rows)', o);
        yd = yd(i);
        yt = yt(i);
        order = order(i);

        end

|

``moments.m``
""""""""""""""""""""""""""

This function summarizes data into a set of moments (used in Step 2 in the procedure on :ref:`home<home>` page).

.. code-block:: console

    output = moments(pos, z, consumer_id, yd, yt)
    
* Inputs: as described above for ``model_seq_search.m``.

* Output: a vector collecting the values of the moments.

..  collapse:: Click to show code <collapse_header>moments.m</collapse_header>

    .. code-block:: matlab
        :class: scrollable-code-block

        function output = moments(pos, z, consumer_id, yd, yt)

        %{

        This function specifies the data moments to be used in NNE.

        Output: 
            A vector collecting the data moments.
        Inputs: 
            pos .. product ranking positions
            z .. other product attributes
            consumer_id .. consumer indices
            yd .. search dummies
            yt .. purchase dummies

        %}

        rows = size(z, 1);

        y = [yd, yt]; % all outcome variables
        x = [z, log(pos)]; % all covariates

        ydn = accumarray(consumer_id, yd); % consumer-level number of searches
        ytn = accumarray(consumer_id, yt); % consumer-level purchase

        y_tilde = [ydn>1, ydn, ytn]; % consumer-level outcomes

        % consumer-level average of x
        x_sum = arrayfun(@(i)accumarray(consumer_id, x(:,i)), 1:size(x,2), 'uni', false);
        x_bar = cell2mat(x_sum)./accumarray(consumer_id, 1);

        % mean vector of y
        m1 = mean(y);

        % cross-covariances between y and x
        m2 = (y - mean(y))'*(x - mean(x))/rows;
        m2 = m2(:)';

        % mean vector of y_tilde
        m3 = mean(y_tilde);

        % cross-covariances between y_tilde and x_bar
        m4 = (y_tilde - mean(y_tilde))'*(x_bar - mean(x_bar))/rows;
        m4 = m4(:)';

        % covariance matrix of y_tilde
        m5 = cov(y_tilde); 
        m5 = m5(tril(true(length(m5))))';

        % collect all moments
        output = [m1, m2, m3, m4, m5];


|

``normalRegressionLayer.m``
""""""""""""""""""""""""""""

This file codes the cross-entropy loss. It extends the Matlab 's built-in MSE loss. This loss function is needed if we want NNE to output estimates of statistical accuracy in addition to point estimates.

..  collapse:: Click to show code <collapse_header>normalRegressionLayer.m</collapse_header>

    .. code-block:: matlab
        :class: scrollable-code-block

        classdef normalRegressionLayer < nnet.layer.RegressionLayer

        %{

        This file codes the loss function for neural net training. It extends the
        Matlab built-in regressionLayer. The regressionLayer uses the MSE loss.
        This file adds a normal cross-entropy loss.

        Set property learn_sd to true to use the cross-entropy loss. In this case,
        the number of neural net outputs doubles to give both the mean and standard
        deviation terms.

        %}

            properties

                learn_sd
                
            end
        
            methods
                
                function layer = normalRegressionLayer(varargin) 

                    p = inputParser;
                    addOptional(p, 'learn_sd', false, @islogical)
                    parse(p, varargin{:})
                    
                    layer.learn_sd = p.Results.learn_sd;

                end

                function loss = forwardLoss(layer, Y, T)
                    
                    if ~ layer.learn_sd

                        Q = 0.5*(Y - T).^2;

                    else

                        k = size(Y,1)/2;
                        
                        S = exp(Y(k+1:2*k, :));
                        V = Y(1:k, :);
                        U = T(1:k, :);
                        
                        Q = log(S) + 0.5*((V - U)./S).^2;

                    end
                    
                    loss = sum(Q(:))/size(Y,2);

                end
                
                function dLdY = backwardLoss(layer, Y, T)
                    
                    if ~ layer.learn_sd

                        dLdY = (Y - T)/size(Y,2);

                    else

                        k = size(Y,1)/2;
                        
                        S = exp(Y(k+1:2*k, :));
                        dS = S;
                        V = Y(1:k, :);
                        U = T(1:k, :);
                        
                        dLdS = 1./S - 1./S.^3.*(V - U).^2;
                        dLdV = (V - U)./S.^2;
                        
                        dLdY = [dLdV; dLdS.*dS]/size(Y,2);

                    end
                end

            end
        end

|

``monte_carlo_data.m``
""""""""""""""""""""""""""

This script generates a dataset of consumer search under a "true" value of the search model parameter, for the purpose of Monte Carlo experiments. It uses the function ``model_seq_search.m`` to simulate the search and purchase choices. The data is saved in a file ``data.mat``.

..  collapse:: Click to show code <collapse_header>monte_carlo_data.m</collapse_header>

    .. code-block:: matlab
        :class: scrollable-code-block

        %{

        This script generates a Monte Carlo data for estimation of the sequential
        search model. The data will be saved in data.mat.

        %}

        clear

        N = 1000; % number of consumers (or search sessions)
        J = 30; % number of options per consumer

        % 1st column are parameter names
        % 2nd column are true parameter value (for Monte Carlo studies).

        set_theta = {  
                    '\beta_1'    0.1     % coefficient (stars)
                    '\beta_2'    0.0     % coefficient (review score)
                    '\beta_3'    0.2     % coefficient (loc score)
                    '\beta_4'   -0.2     % coefficient (chain)
                    '\beta_5'    0.2     % coefficient (promotion)
                    '\beta_6'   -0.2     % coefficient (price)
                    '\eta'       3.0     % outside good
                    '\delta_0'  -4.0     % search cost base
                    '\delta_1'   0.1     % search cost position
                    };

        theta_name = set_theta(:,1)';
        theta_true = cell2mat(set_theta(:,2)');

        rows = N*J;

        % draw the hotel attributes
        z = nan(rows, 6);

        z(:,1) = randsample([2, 3, 4, 5], rows, true, [0.05, 0.25, 0.4, 0.3])'; % star rating
        z(:,2) = randsample([3, 3.5, 4, 4.5, 5], rows, true, [0.08, 0.17, 0.4, 0.3, 0.05])'; % review score
        z(:,3) = normrnd(4, 0.3 ,rows,1); % location score
        z(:,4) = randsample([0, 1], rows, true, [0.2, 0.8])'; % chain hotel dummy
        z(:,5) = randsample([0, 1], rows, true, [0.4, 0.6])'; % promotion dummy
        z(:,6) = normrnd(0.15, 0.6, rows,1); % log price

        % ranking positions
        pos = repmat((1:J)',N,1);

        % consumer index
        consumer_id = repelem(1:N, J)';

        % search and purchase
        curve = importdata('curve_seq_search.csv');
        [yd, yt] = model_seq_search(pos, z, consumer_id, theta_true, curve);

        % save data
        save('data.mat','theta_name','consumer_id','pos','z','yd','yt','N','J')

|

``nne_gen.m``
""""""""""""""""""""""""""

This script generates the training and validation examples (Steps 1 & 2 in the procedure on :ref:`home<home>` page).

* It loads the product attributes (``z`` and ``pos``)  in ``data.mat``.
* It uses ``model_seq_search.m`` to simulate the consumer choices in each training or validation example.
* It uses ``moments.m`` to summarize data in each training or validation example.
* Corner examples (e.g., nobody made a purchase) are dropped.
* At the end, the training and validation examples are saved in a file ``nne_training.mat``.

..  collapse:: Click to show code <collapse_header>nne_gen.m</collapse_header>

    .. code-block:: matlab
        :class: scrollable-code-block

        %{

        This script generates the training and validation examples to be used to
        train NNE. The examples will be saved in nne_trainning.mat.

        Change 'for' to 'parfor' if parallel computing toolbox is available.

        %}

        clear

        %% settings

        % 1st column are parameter names.
        % 2nd and 3rd columns are lower and upper bounds of parameter space Theta.

        Theta = {  
                '\beta_1'   -0.5,   0.5     % coefficient (stars)
                '\beta_2'   -0.5,   0.5     % coefficient (review score)
                '\beta_3'   -0.5,   0.5     % coefficient (loc score)
                '\beta_4'   -0.5,   0.5     % coefficient (chain)
                '\beta_5'   -0.5,   0.5     % coefficient (promotion)
                '\beta_6'   -0.5,   0.5     % coefficient (price)
                '\eta'       2.0,   5.0     % outside good
                '\delta_0'  -5.0,  -2.0     % search cost base
                '\delta_1'  -0.25,  0.25    % search cost position
                };

        label_name = Theta(:,1)';
        lb = cell2mat(Theta(:,2))';
        ub = cell2mat(Theta(:,3))';

        L = 1e4; % number of training & validation examples

        % load reservation utility curve (to be used in search model)
        curve = importdata('curve_seq_search.csv');

        % load observed attributes in data
        load('data.mat', 'pos', 'z', 'consumer_id', 'N', 'J')

        %% generate training & validation examples

        % pre-allocation for training & validation examples
        input = cell(L,1);
        label = cell(L,1);

        for l = 1:L
            
            % draw the value for the search model parameter
            theta = unifrnd(lb, ub);

            % simulate search and purchase outcomes
            [yd, yt] = model_seq_search(pos, z, consumer_id, theta, curve);
            
            % drop corner cases

            buy_rate = sum(yt)/N; % fraction of consumers who purchased
            num_srh  = sum(yd)/N; % number of searches per consumer
            
            if buy_rate > 0 && buy_rate < 1 && num_srh > 1 && num_srh < J

                input{l} = moments(pos, z, consumer_id, yd, yt);
                label{l} = theta;

            end

        end

        % convert cells to matrices
        input = cell2mat(input);
        label = cell2mat(label);

        %% training-validation split

        L = size(input,1); % number of examples excluding corner cases
        L_train = floor(L*0.9); % number of training examples (90-10 split)

        input_train = input(1:L_train,:);
        label_train = label(1:L_train,:);

        input_val = input(L_train+1:L,:);
        label_val = label(L_train+1:L,:);

        %% save

        save('nne_training.mat','input_train','label_train','input_val','label_val','label_name')

|

``nne_train.m``
""""""""""""""""""""""""""

This script trains a shallow neural net (Steps 3 & 4 in the procedure on :ref:`home<home>` page).

* It loads the training and validation examples from ``nne_training.mat`` (created by ``nne_gen.m``).
* It uses ``normalRegressionLayer.m`` for the cross-entropy loss.
* Validation loss is reported. You can use it to choose hyperparameters, such as the number of hidden nodes.
* At the end, it applies the trained neural net to the data in ``data.mat`` and reports the estimate.

..  collapse:: Click to show code <collapse_header>nne_train.m</collapse_header>

    .. code-block:: matlab
        :class: scrollable-code-block

        %{

        This script trains the neural net in NNE, and then applies the trained
        neural net on data.mat to obtain a parameter estimate.

        %}

        clear

        %% settings

        num_nodes = 64; % number of hidden nodes (in shallow neural net)
        learn_sd = true; % whether to learn estimates of statistical accuracy

        %% load training & validation examples

        load('nne_training.mat')

        L_train = size(input_train, 1); % number of training examples
        L_val   = size(input_val,   1); % number of validation examples

        dim_input = size(input_train, 2); % number of inputs by neural net
        dim_label = size(label_train, 2); % number of parameters

        % extend neural net outputs in the case of learn_sd = true  
        output_train = [label_train, zeros(L_train, dim_label*learn_sd)];
        output_val   = [label_val,   zeros(L_val,   dim_label*learn_sd)]; 

        dim_output = size(output_train, 2); % number of outputs by neural net

        %% train a neural net

        opts = trainingOptions( 'adam', ...
                                'ExecutionEnvironment','cpu',...
                                'LearnRateSchedule','piecewise', ...
                                'LearnRateDropPeriod', 200, ...
                                'InitialLearnRate' , 0.01, ...
                                'GradientThreshold', 1,...
                                'MaxEpochs', 300, ...
                                'Shuffle','every-epoch',...
                                'MiniBatchSize', 500,...
                                'L2Regularization', 0, ...
                                'Plots','none', ...
                                'Verbose', true, ...
                                'VerboseFrequency', 500, ...
                                'ValidationData', {input_val, output_val}, ...
                                'ValidationFrequency', 500);

        layers = [  featureInputLayer(dim_input, 'normalization', 'rescale-symmetric')
                    fullyConnectedLayer(num_nodes)
                    reluLayer
                    fullyConnectedLayer(dim_output)
                    normalRegressionLayer('learn_sd', learn_sd)
                    ];

        [net, info] = trainNetwork(input_train, output_train, layers, opts);

        disp(" ")
        disp("Final validation loss is: " + info.FinalValidationLoss)

        %% display figure: estimate vs. truth in validation

        pred_val = predict(net, input_val, exec='cpu');

        figure
        sgtitle('Estimate vs. Truth in Validation')

        p = round(sqrt(dim_label));

        for i = 1:dim_label
            subplot(p, p+1, i)
            scatter(label_val(:,i), pred_val(:,i), '.')
            xlabel(label_name(i))
            axis equal
        end

        %% apply the trained neural net to data.mat

        load('data.mat'); % load data for estimation

        input = moments(pos, z, consumer_id, yd, yt); % calculate data moments to be used as neural net input
        pred = predict(net, input, exec='cpu'); % apply the trained neural net

        estimate = pred(1:dim_label)'; % get point estimate
        sd = exp(pred(dim_label+1:end))'; % get estimate of statistical accuracy
        sd = [sd; nan(dim_label*~learn_sd, 1)]; % fill sd with nan if learn_sd=0

        % display estimates
        result = table(estimate, sd, 'row', label_name, 'var', {'Estimate','SD'});
        result = rmmissing(result, 2);
        disp(" ")
        disp(result)

|

|

