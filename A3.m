%% Assignment 3%%
%%%%%%%%%%%%%%%%%

%Load data
[Xt, Xv, Xtest, Yt, Yv, Ytest] = LoadData();

%Set parameters
Gdparams.eta = 0.184541;
lambda=0.000003;
Gdparams.n_batch=100;
Gdparams.n_epochs=15;
networkShape = [50,30]; %size of hidden layers

%Initialize weights
[W,b] = InitializeParameters(networkShape);

%Run the gradient descent algorithm
[Wstar1,bstar1,acc1,costval1,costtrain1,mu_avg,v_avg] = GD( Xt,Xv,Xtest,Yt,Yv,Ytest,Gdparams,W,b,lambda);
%% Test Learning Rates
Gdparams.n_epochs = 10;
networkShape = [50];
[W,b] = InitializeParameters(networkShape);
%small
Gdparams.eta = 0.01;
lambda=0.000003;
[Wstar1,bstar1,acc1,costval1,costtrain1,mu_avg1,v_avg1] = GD( Xt,Xv,Xtest,Yt,Yv,Ytest,Gdparams,W,b,lambda);
%medium
Gdparams.eta = 0.1;
[Wstar2,bstar2,acc2,costval2,costtrain2,mu_avg2,v_avg2] = GD( Xt,Xv,Xtest,Yt,Yv,Ytest,Gdparams,W,b,lambda);
%large
Gdparams.eta = 1;
[Wstar3,bstar3,acc3,costval3,costtrain3,mu_avg3,v_avg3] = GD( Xt,Xv,Xtest,Yt,Yv,Ytest,Gdparams,W,b,lambda);

%% Parameter Search

Gdparams.n_batch=100;
Gdparams.n_epochs=10;

%Range of learning rate
emin=log10(0.08);
emax=log10(0.2);

%Range of regularization
lmin=log10(10^(-6));
lmax=log10(5*10^(-4));

%File to save results
fileID = fopen('scores.txt','w');
fprintf(fileID,'%6s %12s %12s \n','eta','lambda','acc');


%Run a lot of networks and save the results
for i=1:25
    i
    e = emin + (emax - emin)*rand(1, 1); 
    Gdparams.eta = 10^e;
    l = lmin + (lmax - lmin)*rand(1, 1); 
    lambda = 10^l;
    [Wcell,bcell]=InitializeParameters([50,30]);

    [Wstar,bstar,acc,costval,costtrain1] = GD(Xt,Xv,Xtest,Yt,Yv,Ytest,Gdparams,Wcell,bcell,lambda);
    fprintf(fileID,'%6f %12f %12f\n',Gdparams.eta,lambda,max(acc));
    
end
fclose(fileID);

%%
function[grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda)
% Gradient computation
    [s,shat,mu,v,h,p] = EvaluateClassifier(W,b,X);
    [m,n] = size(p);
    k = length(W);
    g = {};
    for i=1:k
        grad_W{i}=zeros(size(W{i}));
        grad_b{i}=zeros(size(b{i}));
    end
    
    for i = 1:n
       gi = -(Y(:,i)-p(:,i))';
       g = [g,gi];
       grad_b{k} = grad_b{k} + gi';
       grad_W{k} = grad_W{k} + gi'*h{k-1}(:,i)';
       g{i} = g{i}*W{k};
       g{i} = g{i}*diag(shat{k-1}(:,i)>0);
    end
    
    for l = k-1:-1:1
       g = BatchNormPass(g,s{l},mu{l},v{l});
       if l>1
           for i = 1:n
               grad_b{l} = grad_b{l} + g{i}';
               grad_W{l} = grad_W{l} + g{i}'*h{l-1}(:,i)';
               g{i} = g{i}*W{l};
               g{i} = g{i}*diag(shat{l-1}(:,i)>0);
           end
       else
           for i = 1:n
               grad_b{l} = grad_b{l} + g{i}';
               grad_W{l} = grad_W{l} + g{i}'*X(:,i)';
           end
       end    
    end
    
    for j=1:k
       grad_W{j}=grad_W{j}/n+2*lambda*W{j};
       grad_b{j}=grad_b{j}/n;
    end
end


function[G] = BatchNormPass(Gpre, S, Mu, Vb)
%dJ/ds
    [m,n] = size(S);
    epsilon = 10^(-9);
    Vb = diag(Vb + epsilon);
    G={};
    dvb = 0;
    dub = 0;
    for j=1:n
        dvb = dvb + -0.5*Gpre{j}*Vb^(-3/2)*diag(S(:,j)-Mu);
        dub = dub + -Gpre{j}*Vb^(-0.5);
    end
    for i=1:n
        G{i} = Gpre{i}*Vb^(-0.5) + (2/n)*dvb*diag(S(:,i)-Mu)+dub/n; 
    end

end

function[s,shat,mu,v,h,p] = EvaluateClassifier(W,b,X,Mu,V)
%Forward pass of the neural network
%Returns scores from all layers aswell as means and variances.
    k = length(W);
    [m,n] = size(X);
    s = {};
    h = {};
    p = [];
    mu = {};
    v = {};
    shat = {};
    for l = 1:k-1
       if l == 1
           s{1} = W{1}*X + b{1};
           if nargin < 4
               mu{1} = mean(s{1},2);
               v{1} = var(s{1},0,2)*(n-1)/n;
           else
               mu{1} = Mu{1};
               v{1} = V{1};
           end
           shat{1} = BatchNormalize(s{1}, mu{1}, v{1});
           h{1} = max(shat{1},0);
           
       else
          s{l} = W{l}*h{l-1} + b{l};
          if nargin <4
                mu{l} = mean(s{l},2);
                v{l} = var(s{l},0,2)*(n-1)/n;
          else
              mu{l} = Mu{l};
              v{l} = V{l};
          end
          shat{l} = BatchNormalize(s{l}, mu{l}, v{l});
          h{l} = max(0,shat{l});
       end
    end
    if k == 1
       sFinal = W{1}*X + b{1}; 
    else
        sFinal = W{end}*h{end} + b{end};
    end
    for i = 1:n
        p(:,i) = exp(sFinal(:,i))/(ones(1,10)*exp(sFinal(:,i)));
    end
end

function [ Wstar,bstar,acc,costval,costtrain,mu_avg,v_avg] = GD( Xt,Xv,Xtest,Yt,Yv,Ytest,Gdparams,W,b,lambda)
    %Gradient descent with momentum

    acc=[];
    costval=[];
    costtrain=[];
    k=length(W);
    vW={};
    vb={};
    for j=1:k
        vW=[vW,zeros(size(W{j}))];
        vb=[vb,zeros(size(b{j}))];
    end
    rho=0.9;
    alpha = 0.99;
    [c,Ntrain]=size(Xt);
    
    for y=1:Gdparams.n_epochs
       if y == 1
          [~,~,mu_avg,v_avg,~,~] = EvaluateClassifier(W,b,Xt(:,1:100)); 
       end

       
       [~,~,~,~,~,Pt] = EvaluateClassifier(W,b,Xt);
       [~,~,~,~,~,Pv] = EvaluateClassifier(W,b,Xv);
       [~,ht,~,~,~,Ptest] = EvaluateClassifier(W,b,Xtest,mu_avg,v_avg);
       
       
       [accuracy] = ComputeAccuracy(Ptest,Ytest);
       [trainLoss,~] = ComputeCost(W,Pt,Yt,lambda);
       [valLoss,~] = ComputeCost(W,Pv,Yv,lambda);
       
       costtrain = [costtrain, trainLoss];
       costval = [costval, valLoss];
       acc = [acc, accuracy];
       for j=1:Ntrain/Gdparams.n_batch
            j_start = (j-1)*Gdparams.n_batch + 1;
            j_end = j*Gdparams.n_batch;
            inds = j_start:j; 
            Xbatch = Xt(:,j_start:j_end);
            Xbatch = Xbatch+0.12*randn(size(Xbatch)); 
            Ybatch = Yt(:,j_start:j_end);
            [grad_W,grad_b]=ComputeGradients(Xbatch,Ybatch,W,b,lambda);
            [~,~,mu0,v0,~,~] = EvaluateClassifier(W,b,Xbatch);
            for g=1:k
                vW{g}=rho*vW{g}+Gdparams.eta*grad_W{g};
                vb{g}=rho*vb{g}+Gdparams.eta*grad_b{g};
                W{g}=W{g}-vW{g};
                b{g}=b{g}-vb{g};
                if g ~=k
                    mu_avg{g} = alpha*mu_avg{g} + (1-alpha)*mu0{g};
                    v_avg{g} = alpha*v_avg{g} + (1-alpha)*v0{g};
                end
            end
       end
       Gdparams.eta = Gdparams.eta*0.8;
    end
    Wstar = W;
    bstar = b;
end





function[loss,o] = ComputeCost(W,P,Y,lambda)
o = 0;
%Computes loss of network
    [n,m]=size(P);
    J=0;
    for i=1:m
       currJ=-log(Y(:,i)'*P(:,i));
       J=J+currJ;
    end
    len=length(W);
    lambdacost=0;
    for i=1:len
        
        lambdacost=lambdacost + sum(sum(W{i}.*W{i}));
    end
    J=J/m+lambda*lambdacost;
    loss = J;
end



function[accuracy] = ComputeAccuracy(P,Y)
%Computes Accuracy of estimates P
    [n,m]=size(P);
    correctGuesses=0;
    for i=1:m
        [maxP,indP]=max(P(:,i));
        [maxY,indY]=max(Y(:,i));
        if indP==indY
           correctGuesses=correctGuesses+1; 
        end
    end
    accuracy=correctGuesses/m;
end




function [shat] = BatchNormalize(s,u,v)
    epsilon = 10^(-9);
    shat = diag(v+epsilon)^(-0.5)*(s-u);
end


function [ Wcell,bcell ] = InitializeParameters(network_shape)
% Initilizes the weights
    network_shape=[3072, network_shape, 10];
    Wcell={};
    bcell={};
    len=length(network_shape);
    mu = 0;
    sigma = 0.001;
    for i = 1:len-1
       Wn=normrnd(mu,sigma,[network_shape(i+1),network_shape(i)]);
       %Wn = ones([network_shape(i+1),network_shape(i)]);
       Wcell=[Wcell, Wn];
       
       bn=zeros([network_shape(i+1),1]);
       %bn = ones([network_shape(i+1),1]);
       bcell=[bcell,bn];
    end
end

function [Xt, Xv, Xtrain, Yt, Yv, Ytrain] = LoadData()
    [Xt, Yt] = LoadBatch('data_batch_1.mat');
    [Xv, Yv] = LoadBatch('data_batch_2.mat');
    [Xtrain, Ytrain] = LoadBatch('test_batch.mat');
    
end


function [ X,Y,y,m ] = LoadBatch(filename)
% Loads data from file
    addpath Datasets/cifar-10-batches-mat/;
    A = load(filename);
    X=double(A.data')/255;
    max(max(X));
    m=mean(X,2);
    X=X-repmat(m, [1, size(X, 2)]);
    y=double(A.labels')+ones(size(A.labels'));
    Y=full(ind2vec(y));
end




function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

%[c, ~] = ComputeCost(X, Y, W, b, lambda);
[s11,~,~,~,h11,p] = EvaluateClassifier(W,b,X);
[c, ~] = ComputeCost(W, p, Y, lambda);
for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        [s11,~,~,~,h11,p] = EvaluateClassifier(W,b_try,X);
        [c2, ~] = ComputeCost(W, p, Y, lambda);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    %numel(W{j})
    for i=1:numel(W{j})
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        
        [s11,~,~,~,h11,p] = EvaluateClassifier(W_try,b,X);
        [c2, ~] = ComputeCost(W_try, p, Y, lambda);
        %[c2, ~] = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{j}(i) = (c2-c) / h;
    end
end


end

function plotAccuracy(acc)
    set(groot,'defaulttextinterpreter','latex');  
    set(groot, 'defaultAxesTickLabelInterpreter','latex');  
    set(groot, 'defaultLegendInterpreter','latex');
    ep = 1:length(acc);
    plot(ep, acc, 'b--','linewidth',3);
    legend('Test Accuracy')
    axis([1,length(acc), min(acc), max(acc)]);
    set(gca,'fontsize', 20)
    xlabel('Epochs')
    ylabel('Test Accuracy')
end

function plotCost(costval,costtrain)
    set(groot,'defaulttextinterpreter','latex');  
    set(groot, 'defaultAxesTickLabelInterpreter','latex');  
    set(groot, 'defaultLegendInterpreter','latex');
    ep = 1:length(costtrain);
    plot(ep, costval, 'b--', ep, costtrain, 'r--', 'linewidth',3)
    legend('Validation Loss', 'Training Loss')
    axis([1,length(costtrain), min(costtrain), max(costtrain)]);
    set(gca,'fontsize', 20)
    xlabel('Epochs')
    ylabel('Cross Entropy Loss')
end

