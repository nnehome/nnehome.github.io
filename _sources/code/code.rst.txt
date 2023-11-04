:html_theme.sidebar_secondary.remove:

:parenttoc: True

.. _code:

Code
============

All Matlab codes can be found on the `GitHub <https://github.com/nnehome/nne-matlab>`_ page. 
On the home page, we describe the two key functions ``nne_gen.m`` and ``nne_train.m`` to implement NNE. 
On this page, we describe the other accompanying functions and files used in the estimation.

NNE estimation
--------------

``Moments.m``
"""""""""""""""""

This function summarizes data into data moments which serve as inputs into neural net estimation. It can be adapted to other structural models.

.. code-block:: matlab
    :class: scrollable-code-block

    function output = Moments(pos, z, consumer_id, yd, yt, type)

    n = size(z,1);

    y = [yd, yt]; % outcome; search+purchase
    x = [z, log(pos)]; % covariates + position

    ydn = accumarray(consumer_id, yd);  % number of searches
    ytn = accumarray(consumer_id, yt);  % whether bought inside goods

    y_tilde = [ydn>1, ydn, ytn];

    x_tilde = splitapply(@mean,x,consumer_id);

    % if not specified, use type 4, which is the main specification
    if exist('type','var')==0

        type = 4;

    end


    if type==1 
        
        m1 = mean(y);
        m2 = (y - mean(y))'*(x - mean(x))/n; m2 = m2(:)';
        output = [m1, m2];

    elseif type==2

        m1 = mean(y);
        m2 = (y - mean(y))'*(x - mean(x))/n; m2 = m2(:)';
        m3 = mean([ydn, ytn]);
        m4 = ([ydn, ytn] - mean([ydn, ytn]))'*(x_tilde - mean(x_tilde))/n; m4 = m4(:)';
        output = [m1, m2, m3, m4];

    elseif type==3

        m1 = mean(y);
        m2 = (y - mean(y))'*(x - mean(x))/n; m2 = m2(:)';
        m3 = mean(y_tilde);
        m4 = (y_tilde - mean(y_tilde))'*(x_tilde - mean(x_tilde))/n; m4 = m4(:)';
        output = [m1, m2, m3, m4];
    
    elseif type==4 % main specification

        m1 = mean(y);
        m2 = (y - mean(y))'*(x - mean(x))/n; m2 = m2(:)';
        m3 = mean(y_tilde);
        m4 = (y_tilde - mean(y_tilde))'*(x_tilde - mean(x_tilde))/n; m4 = m4(:)';
        m5 = cov(y_tilde); m5 = m5(tril(true(length(m5))))';
        output = [m1, m2, m3, m4, m5];

    elseif type==5

        m1 = mean(y);
        m2 = (y - mean(y))'*(x - mean(x))/n; m2 = m2(:)';
        m3 = mean(y_tilde);
        m4 = (y_tilde - mean(y_tilde))'*(x_tilde - mean(x_tilde))/n; m4 = m4(:)';
        m5 = cov(y_tilde); m5 = m5(tril(true(length(m5))))';
        m6 = (y - mean(y))'*(x.^2 - mean(x.^2))/n; m6 = m6(:)';
        output = [m1, m2, m3, m4, m5, m6];

    elseif type==6

        m1 = mean(y);
        m2 = (y - mean(y))'*(x - mean(x))/n; m2 = m2(:)';
        m3 = mean(y_tilde);
        m4 = (y_tilde - mean(y_tilde))'*(x_tilde - mean(x_tilde))/n; m4 = m4(:)';
        m5 = cov(y_tilde); m5 = m5(tril(true(length(m5))))';
        m6 = (y - mean(y))'*(x.^2 - mean(x.^2))/n; m6 = m6(:)';
        m7 = (y_tilde - mean(y_tilde))'*(x_tilde.^2 - mean(x_tilde.^2))/n; m7 = m7(:)';
        output = [m1, m2, m3, m4, m5, m6, m7];
    end

    end


inputs:

-	``pos``: ranking position of the option
-	``z``: product observables
-	``consumer_id``: consumer id 
-	``yd``: dummy variable denoting consumer clicking outcome
-	``yt``: dummy variable denoting consumer purchase outcome
-	``type``: denote the different type of moments. Type=4 is the main specification. Type 1 to 6 calculates 16,32,40,46,60 and 81 number of moments (as in Table 5 of the paper).

|

``PredSummary.m`` 
"""""""""""""""""

This function calculates the bias and rmse of the test data after training the neural network model.

.. code-block:: matlab
    :class: scrollable-code-block

    % measure misclassifiction rate


    function err  = PredSummary(input_test, label_test, label_name, net, varargin)

    option = inputParser;
    option.addParameter('figure', 1);
    option.addParameter('table', 1);
    option.parse(varargin{:})

    n = size(label_test,1);
    k = size(label_test,2);
    p = numel(label_name);

    if p == k
        learn_se = 0;
    elseif 2*p == k
        learn_se = 1;
    else
        error('label format not recognized,')
    end

    y_hat = predict(net, input_test, 'acceleration', 'none','ExecutionEnvironment','cpu');
    err = y_hat(:,1:p) - label_test(:,1:p);

    if learn_se == 1
        y_hat(:, p+1:k) = PositiveTransform(y_hat(:,p+1:k));
        sdd = y_hat(:, p+1:k);
    end

    set(0,'DefaultTextInterpreter','latex')
    set(groot, 'defaultLegendInterpreter','latex','defaultAxesTickLabelInterpreter','latex')

    q = min(10, p);

    %% plot histogram

    if option.Results.figure
        
        figure('Position', [100 500 250*q 420])
        
        for j = 1:q
            
            subplot(2, q, q+j)
            histogram(err(:,j), 'EdgeColor', 'none', 'FaceColor', [0 0.4 0.7])
            xlabel("$\widehat{" + label_name{j} + "}-" + label_name{j} + "$")
            axis tight
            ylim(ylim*1.1);
            
        end
        
        for j = 1:q
            
            subplot(2, q, j)
            scatter(label_test(:,j), err(:,j) + label_test(:,j), 30, [0 0.4 0.7], '.')
            line45 = refline(1,0);
            line45.Color = 'r';
            ylabel({" ";" ";"$\widehat{" + label_name{j} + "}$"})
            xlabel("$" + label_name{j} + "$")
            axis tight
            box on
            
        end
        
    end

    %% print result table

    if option.Results.table

        bias = num2str(mean(err)',3)+" ("+num2str(std(err)'/sqrt(n),1)+")";
        rmse = num2str(sqrt(mean(err.^2)'),3)+" ("+num2str(.5./sqrt(mean(err.^2)').*std(err.^2)'/sqrt(n),1)+")";
        
        if learn_se == 0
            mean_SD = num2str(nan(p,1));
        else
            mean_SD = num2str(mean(sdd)',3)+" ("+num2str(std(sdd)'/sqrt(n),1)+")";
        end
        
        bias = string(bias);
        rmse = string(rmse);
        mean_SD = string(mean_SD);
        
        result = table(bias, rmse, mean_SD, 'RowNames', label_name);
        disp(result)
        
    end


inputs:

-	``input_test``: moment input in the test set
-	``label_test``: label output in the test set
-	``label_name``: name of the label
-	``net``: trained neural net
-	``figure`` (1/0): optional input to show figure
-	``table`` (1/0): optional input to show table

|

``Statistics.m`` 
""""""""""""""""

This function calculates the summary statistics of the search data.

.. code-block:: matlab
    :class: scrollable-code-block

    function [buy_rate, search_rate, num_search, pos_search] = Statistics(yd, yt, pos, consumer_id, display)

    % index = find(X(:,end));

    ydn = accumarray(consumer_id, yd);
    ytn = accumarray(consumer_id, yt);

    buy_rate = mean(ytn);
    search_rate = mean( ydn > 1);
    num_search = mean(ydn);
    search_position = pos(yd==1);
    pos_search = mean(search_position);

    if display
        disp(' ')
        disp("Frequency of buying: " + buy_rate)
        disp("Frequency of search: " + search_rate)
        disp("Average number of searches: " + num_search)
        disp("Average position of searches: " + pos_search)
        
    end
    % accumarray(pos, yd)
    % tabulate(ydn)

|

``gen_seq_search.m`` 
""""""""""""""""""""

This function generates outcomes based on observables and error draws. It is specific to sequential search model and can be adapted to other structural models. 
This function is adapted from the replication code of Ursu "The Power of Rankings". 

.. code-block:: matlab
    :class: scrollable-code-block

    function [yd, yt, order] = gen_seq_search(pos, X, consumer_id, theta, eps, eps0, curve)
    %% setup

    % number of options for each consumer
    Ji = accumarray(consumer_id,1);
    Ji = Ji(consumer_id);

    bepos = theta(end); % last par, position
    constc = theta(end - 1); % par (2nd to last): constant in search cost

    v0 = theta(end - 2); % outside option

    rows = length(consumer_id); % number of observations
    N = consumer_id(rows); % number of consumers

    [~, index] = ismember(1:N, consumer_id); % index of consumers in the data

    %form utility from data and param
    eutility = X * theta(1:size(X,2))';

    utility = eutility + eps;

    % utility of the outside option
    u0 = v0 + eps0;

    %search cost
    % c, and therefore m, only changes with pos
    sz = size(curve,1);
    pos_unique = sort(unique(pos));
    m_pos = zeros(length(pos_unique),1);
    for i = 1:length(pos_unique)
    %     c_i = exp(constc + i.*bepos);
        c_i = exp(constc + log(i).*bepos);
        if c_i<curve(1,2) && c_i>=curve(sz,2)
            for n = 2:length(curve)
                if (curve(n,2) == c_i)
                    m_pos(i) = curve(n,1);
                elseif ((curve(n-1,2)>c_i)&& (c_i>curve(n,2)))
                    m_pos(i) = (curve(n,1)+curve(n-1,1))/2;
                end
            end
        elseif c_i>=curve(1,2)
            m_pos(i) = -c_i;
        elseif c_i<curve(sz,2)
            m_pos(i) = 4.001;
        end
    end
    m = m_pos(pos);

    %reservation utilities
    z = m + eutility;

    %order by z for each consumer
    da = [consumer_id, pos, Ji, X, utility, eutility, z];
    whatz = size(da,2);
    whateu = whatz - 1;
    whatu = whateu - 1;

    order = zeros(rows,1);

    for m = 1:N
        n = index(m);
        J = Ji(n);
    %     for j = n:n+J-1
        [~, order(n:n+J-1)] = sort(da(n:n+J-1, whatz),'descend');
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

    %for next click decisions: if z is higher than outside
    %option and higher than all utilities so far, then increase click d by one
    % It is ok to do this because we ordered the z's first, so we know that zn>zn+1
    for i = 1:N
        J = Ji(index(i));
        for j = 1:(J-1)
            ma = max(data(index(i):(index(i)+j-1), whatu), u0(i));
            if data(index(i)+j, whatz) > ma
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





input: 

-	``pos``: ranking position of the option
-	``X``: product observables
-	``consumer_id``: consumer id 
-	``theta``: parameter value
-	``eps``: error draw for the utility of the options
-	``eps0``: error draw for the outside option
-	``curve``: a table that stores the standardized search cost and the corresponding reservation utility

output: 

-	``yd``: dummy variable denoting consumer clicking outcome
-	``yt``: dummy variable denoting consumer purchase outcome
-	``order``: order of search

|

``normalRegressionLayer.m`` 
"""""""""""""""""""""""""""

This function is a custom neural net layer which defines the loss function used in neural net training. It is used as the last layer of the neural net ``layers``. 
``forwardLoss`` function specifies the loss between predictions and the training targets. Two types of loss function can be used in NNE. 
If ``learn_standard_error = false``, NNE uses the mean squared error loss (``C_1(f)`` in Eq 5 in the paper). If ``learn_standard_error = true``, NNE uses the cross-entropy 
loss (``C_2(f)`` in Eq 6 in the paper). ``backwardLoss`` function specifies the derivative of the loss with respect to the predictions. When ``learn_standard_error = true``, 
NNE predicts both the point estimates and the standard deviations. 

.. code-block:: matlab
    :class: scrollable-code-block

    classdef normalRegressionLayer < nnet.layer.RegressionLayer
            
        properties
            % (Optional) Layer properties.
            
            Simple
            
            % Layer properties go here.
        end
    
        methods
            
            function layer = normalRegressionLayer(varargin) 

                p = inputParser;
                addOptional(p, 'simple', false, @islogical)
                parse(p, varargin{:})
                
                layer.Simple = p.Results.simple;

            end

            function loss = forwardLoss(layer, Y, T)
                
                if layer.Simple

                    Q = 0.5*(Y - T).^2;
                    
                else

                    k = size(Y,1)/2;
                    
                    [S, ~] = PositiveTransform(Y(k+1:2*k, :));
                    U = Y(1:k, :);
                    X = T(1:k, :);
                    
                    Q = log(S) + 0.5*((U - X)./S).^2;

                end
                
                loss = sum(Q(:))/size(Y,2);

            end
            
            function dLdY = backwardLoss(layer, Y, T)
                
                if layer.Simple

                    dLdY = (Y - T)/size(Y,2);

                else
                    k = size(Y,1)/2;
                    
                    [S, dS] = PositiveTransform(Y(k+1:2*k, :));
                    U = Y(1:k, :);
                    X = T(1:k, :);
                    
                    dLdS = 1./S - 1./S.^3.*(U - X).^2;
                    dLdU = (U - X)./S.^2;
                    
                    dLdY = [dLdU; dLdS.*dS]/size(Y,2);

                end
            end

        end
    end

|

``PositiveTransform.m``
"""""""""""""""""""""""

This function is used in ``normalRegressionLayer.m`` to transform to positive numbers for the variance terms.

.. code-block:: matlab
    :class: scrollable-code-block

    function [value, derivative] = PositiveTransform(x)

    value = exp(x);
    derivative = value;

|

``set_up.m`` 
""""""""""""
This function generates the Monte Carlo data for sequential search models, which is not needed when estimating a real dataset. 
It also defines the bounds from which to draw :math:`\theta^{(l)}`. 

.. code-block:: matlab
    :class: scrollable-code-block

    %% config

    N = 1000; % number of consumers
    J = 30; % options per consumer

    % z dim = 7, the same as the empirical data
    bounds = {  
                0.1,   -0.5,  0.5,	'\beta_1'   	% coefficient (stars)
                0.0,   -0.5,  0.5, 	'\beta_2'   	% coefficient (review score)
                0.2,   -0.5,  0.5, 	'\beta_3'   	% coefficient (loc score)
            -0.2,   -0.5,  0.5, 	'\beta_4'       % coefficient (chain)
                0.2,   -0.5,  0.5,  '\beta_5'   	% coefficient (promotion)
            -0.2,   -0.5,  0.5,  '\beta_6'     	% coefficient (price)
                3.0,    2.0,  5.0, 	'\eta'          % outside good
            -4.0,   -5.0,  -2.0,	'\delta_0'      % search cost base
                0.1,   -0.25,  0.25,'\delta_1'  % search cost position
            };

    theta_true = cell2mat(bounds(:,1))';
    lb = cell2mat(bounds(:,2))';
    ub = cell2mat(bounds(:,3))';
    label_name = bounds(:,4)';

    %% simulate data

    curve = importdata('tableData.csv');

    rows = N*J;

    outside = false(N*J,1);
    outside(1:J:N*J) = 1;

    % draw the hotel characteristics z
    z = [randsample([2, 3, 4, 5], rows, true, [0.05, 0.25, 0.4, 0.3])',...
        randsample([3, 3.5, 4, 4.5, 5], rows, true, [0.08, 0.17, 0.4, 0.3, 0.05])',...
        4 + 0.3*randn(rows,1),...
        randsample([0, 1], rows, true, [0.2, 0.8])',...
        randsample([0, 1], rows, true, [0.4, 0.6])',...
        (0.15 + 0.6*randn(rows,1))];

    pos = repmat((1:J)',N,1);
    consumer_id = cumsum(outside);
    index = find(z(:,end));

    %draw eps for each consumer-firm combination
    eps = randn(length(consumer_id),1);

    %draw eps for outside option
    eps0 = randn(length(unique(consumer_id)),1);

    [yd, yt] = gen_seq_search(pos, z, consumer_id, theta_true, eps, eps0, curve);

    Statistics(yd, yt, pos, consumer_id, true);

    %% save

    save('data.mat','label_name','theta_true','lb','ub','consumer_id','pos','z','yd','yt')

|

``tableData.csv``
"""""""""""""""""

This file stores the standardized search cost and the corresponding reservation utility. 
This is specific to the estimation of sequential search model.

|

SMLE estimation
---------------

``tableData.csv``, ``gen_seq_search.m``, ``set_up.m``, ``Statistics.m`` are defined the same as in NNE estimation.

``liklOutsideFE.m``
"""""""""""""""""""

This function calculates the log-likelihood for one consumer using the smoothed likelihood approach. 
The function is adapted from the replication code of Ursu "The Power of Rankings".

.. code-block:: matlab
    :class: scrollable-code-block

    % the likelihood function is adapted from the replication code of Ursu "The Power of Rankings"

    function loglik = liklOutsideFE(be, pos, X, consumer_id, yd, yt, R, w, eps_draw,eps0_draw,curve)

    bepos = be(end); % last par, position
    constc = be(end - 1); % 2nd to last par: constant in search cost
    v0 = be(end - 2); % outside option

    Ji = accumarray(consumer_id,1); % number of options per consumer (size N by 1)

    rows = length(consumer_id);
    N = consumer_id(rows);

    % index of consumers in the data (size N by 1)
    index = [1; find(ischange(consumer_id))];

    % form utility from data and param
    eutility = X * be(1:size(X,2))';
    utility = repmat(eutility, [1,R]) + eps_draw;

    % utility of the outside option
    u0 = v0 + eps0_draw;

    % search cost & reservation utility
    sz = size(curve,1);
    pos_unique = sort(unique(pos));
    m_pos = zeros(length(pos_unique),1);
    for i = 1:length(pos_unique)
        c_i = exp(constc + log(i).*bepos); % search cost
        if c_i<curve(1,2) && c_i>=curve(sz,2)
            for n = 2:length(curve)
                if (curve(n,2) == c_i)
                    m_pos(i) = curve(n,1);
                elseif ((curve(n-1,2)>c_i)&& (c_i>curve(n,2)))
                    m_pos(i) = curve(n,1) + (c_i - curve(n,2))/(curve(n-1,2)-curve(n,2))...
                        *(curve(n-1,1) - curve(n,1));
                end
            end
        elseif c_i>=curve(1,2)
            m_pos(i) = -c_i;
        elseif c_i<curve(sz,2)
            m_pos(i) = 4.001;
        end
    end
    m = m_pos(pos);

    %reservation utilities
    z = m + eutility;

    %%%% LIKELIHOOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % idea: denominator = 1 + sum(w_click + w_tran)
    % probability prob = 1 / denominator

    % w_click
    denomd = zeros(rows,R);
    for r = 1:R
        for i = 1:N
            n = index(i);
            J = Ji(i);
            % find entry s where last click occurs for a consumer
            s = find(yd(n:n+J-1),1,'last');
            % if consumer has at least one click (s>1; one free search)
            if (s > 1 && s < J)
                % compute ma = max utility of those searchd (incl. outside option)
                ma = max([utility(n:n+s-1,r); u0(i,r)]);
                % continue to search condition
                %1. z_searched(i) > max{u_searched(i-1), u_outside}
                for l = n+1:n+s-1
                    denomd(l,r) = exp(w*(z(l) - u0(i,r)));
                    k = n;
                    while k<=l-1
                        denomd(l,r) = denomd(l,r) + exp(w*(z(l) - utility(k,r)));
                        k = k + 1;
                    end
                end
                %  stopping rules
                %2. max u_searched > z_notsearched
                for l = n+s:n+J-1
                    denomd(l,r) = exp(w*(ma - z(l)));
                end
            elseif (s > 1 && s == J)
                % continue to search
                for l = n+1:n+s-1
                    denomd(l,r) = exp(w*(z(l) - u0(i,r)));
                    k = n;
                    while k<=l-1
                        denomd(l,r) = denomd(l,r) + exp(w*(z(l) - utility(k,r)));
                        k = k + 1;
                    end
                end
            elseif s==1
                % if there is only one free search
                % max(u_outside, u1) > all other z's
                for l = n+1:n+J-1
                    denomd(l,r) = exp(w*(max([utility(n:n+s-1,r); u0(i,r)]) - z(l)));
                end
            end
        end
    end


    %w_tran
    denomt = zeros(rows,R);
    for r = 1:R
        for m = 1:N
            n = index(m);
            J = Ji(m);
            %find index of tran st and of last click
            st = find(yt(n:n+J-1),1,'last');
            sd = find(yd(n:n+J-1),1,'last');
            kt = n;
            if isempty(st) % no purchase
                while kt <= n+sd-1
                    % outside option is better than all clicked options
                    denomt(n,r) = denomt(n,r) + exp(w*(u0(m,r) - utility(kt,r)));
                    kt = kt+1;
                end
            else % purchase option st
                % purchased option is better than outside option
                denomt(n+st-1,r) = denomt(n+st-1,r) + exp(w*(utility(n+st-1,r) - u0(m,r)));
                while kt<=n+sd-1
                    % purchased option is better than all other clicked options
                    if kt~=n+st-1
                        denomt(n+st-1,r) = denomt(n+st-1,r) + exp(w*(utility(n+st-1,r) - utility(kt,r)));
                    end
                    kt = kt+1;
                end
            end
        end
    end

    %add up search and tran partial denoms: add w_tran and w_click up
    den = denomd + denomt;

    denfull = zeros(N,R);
    for r = 1:R
        denfull(:,r) = 1 + accumarray(consumer_id, den(:,r));
    end

    %probability
    prob = 1./denfull;
    %likelihood
    loglik = log(mean(prob,2) + 1e-16);


    end

input: 

-	``be``: parameter value
-	``pos``: ranking position of the option
-	``X``: product observables
-	``consumer_id``: consumer id 
-	``yd``: dummy variable denoting consumer clicking outcome
-	``yt``: dummy variable denoting consumer purchase outcome
-	``R``: number of simulations
-	``w``: the negative of the smoothing parameter
-	``eps_draw``: error draw for the utility of the options
-	``eps0_draw``: error draw for the outside option
-	``curve``: a table that stores the standardized search cost and the corresponding reservation utility

|

``Objective_mle.m``
"""""""""""""""""""

This function returns the negative value of the sum of the log-likelihood by calling the ``liklOutsideFE.m`` function.

.. code-block:: matlab
    :class: scrollable-code-block

    function output = Objective_mle(be, pos, X, consumer_id, yd, yt, R, w, eps_draw,eps0_draw,curve) 

    loglik = liklOutsideFE(be, pos, X, consumer_id, yd, yt, R, w, eps_draw,eps0_draw,curve);

    output = -sum(loglik);

    end

|

``main_mle.m``
""""""""""""""

This function implements the simulated maximum likelihood using the smoothed likelihood approach. The smoothing parameter can be set by the ``w`` variable. 

.. code-block:: matlab
    :class: scrollable-code-block

    %% set up

    clear; 
    seed = 1; 
    R = 50; % number of simulations
    w = -7; % smoothing parameter

    tic;

    rng(seed)

    set_up % generate a search dataset, save in data.mat

    load('data.mat')

    % table with (normalized) search cost and reservation utility
    curve = importdata('tableData.csv');

    %% MLE  estimation for Monte Carlo

    %draw eps for each consumer-firm combination
    eps = randn(length(consumer_id),1);
    %draw eps for outside option
    eps0 = randn(length(unique(consumer_id)),1);

    % simulate data
    [yd, yt, order] = gen_seq_search(pos, z, consumer_id, theta_true, eps, eps0, curve);

    data = [consumer_id, pos, z, yd, yt, order];
    data = sortrows(data, [1,-9,11]); % sort by consumer_id and order of clicks

    yd = data(:, end-2);
    yt = data(:, end-1);
    pos = data(:, 2);
    z = data(:,3:(2+size(z,2)));

    %draw eps for each consumer-firm combination
    eps_draw = randn(length(consumer_id),R);
    %draw eps for outside option
    eps0_draw = randn(length(unique(consumer_id)),R);

    %initial parameter vector
    be0 = (ub + lb)/2;

    % options for estimation
    options = optimoptions( 'fmincon',...
        'Display', 'iter',...
        'FinDiffType', 'central',...
        'FunValCheck', 'on',...
        'MaxFunEvals', 1e6,...
        'MaxIter', 1e6);

    % [be,fval,~,output]=fminunc(@Objective_mle,be0,options,pos, X, consumer_id, yd, yt,...
    % R, w, eps_draw,eps0_draw,curve);

    [be,fval,~,output] = fmincon(@Objective_mle,be0,[],[],[],[],lb,ub,[],options,...
        pos, z, consumer_id, yd, yt, R, w, eps_draw,eps0_draw,curve);

    be = reshape(be, [1, length(be)]);

    ll_optim = liklOutsideFE(be, pos, z, consumer_id, yd, yt, R, w,eps_draw,eps0_draw,curve);

    G = zeros(length(ll_optim), length(be));
    for j = 1:length(be)
        par_input = be;
        par_input(j) = par_input(j) + 1e-3;
        ll_j = liklOutsideFE(par_input, pos, z, consumer_id, yd, yt, R, w,eps_draw,eps0_draw,curve);
        G(:, j) = (ll_j - ll_optim)/1e-3;
    end

    se = sqrt(diag(inv(G' * G)));

    se = reshape(se, [1, length(se)]);

    toc;
    mle_time = toc/60;
    
    A = [be se output.funcCount fval mle_time];

    csvwrite(sprintf('theta_R%d_w%d.csv',R,-w), A);

|

