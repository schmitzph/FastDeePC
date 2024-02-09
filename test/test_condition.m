%%%%%%%%% Test preconditioning of BFGS
%%%%%%%%% calculates condition numbers

clear
close all
seed = 123;
rng(seed) % random seed

timestamp = datetime('now');
timestamp.Format = 'yyyy-MM-dd_HHmmss';

addpath('../functions')

% system dimensions
stateDim_range = [10,20,30,40,50,60,70,80,100,120,140,160,180,200,220,240,260,280,300,350,400,450,500]; % caution, for large dimension much memory is used and Matlab tends to segfault
inputDim = 50;

% prediction horizon
L = 50;

% BFGS parameters
maxiter = 3000;
maxcor = 3000;
gtol = 1e-6;
verb = true;

cond1 = zeros(length(stateDim_range),1);
cond2 = zeros(length(stateDim_range),1);

for k = 1:length(stateDim_range)
    % generate system (A,B)
    stateDim = stateDim_range(k);
    fprintf('\n\n\n\n########## New round #######\nstatedim: %i\tinputDim: %i\t L: %i\n', stateDim, inputDim, L);
    [A, B] = spawnSystem(stateDim, inputDim, 0.5, 0.9);

    % p.e. trajectory
    U = peInput(inputDim, L+stateDim, true, true);
    X = calcState(U, A, B);

    % Toeplitz matrix of size (m x n) with depth L represented by the data seq and Lam,
    % respectively
    seq = [X;U];
    [r,N] = size(seq);
    Lam = fft(circshift(seq,-L+1,2), N, 2);
    n = N-L+1;
    m = r*L;

    Q = ones(m,1); % OCP weights
    reg = 0.0; % regularization parameter
    rgtol = gtol * sqrt(stateDim);
    
    % (random) initial conditon x0 and reference stored in w
    ker = null([A-eye(stateDim), B]);
    w = repmat(ker(:,1), L, 1);
    x0 = 2*rand(stateDim,1)-1;
    ind = (m-(stateDim+inputDim)+1:m-inputDim);
    w(ind) = x0;
    
    z0 = 2*rand(N-L+1,1)-1;
    lamb0 = rand(length(ind),1);

    [~, flag, iter, ~, S, Y, rho] = lbfgs(z0,Lam,r,N,L,w,Q,ind,lamb0,maxiter,maxcor,reg,rgtol,verb,false);

    % compute condition numbers
    T = buildToeplitz(seq, L);
    fprintf('\nT built')
    TtQT = T'*(Q.*T);
    fprintf('\nTtQT built')
    clear T;
    s1 = svd(TtQT);
    cond1(k) = s1(1)/s1(L*inputDim+stateDim);
    fprintf('\ncond1 computed')

    for j = 1:n
        TtQT(:,j) = inverse_hesse(TtQT(:,j),iter,maxiter,maxcor,S,Y,rho);
        if mod(j,100)==0
            fprintf('\n%i/%i',j,n)
        end
    end
    fprintf('\npreconditioned TtQt built')
    s2 = svd(TtQT);
    fprintf('\ncond2 computed')
    clear TtQT;
    clear S;
    clear Y;
    clear rho;
    cond2(k) = s2(1)/s2(L*inputDim+stateDim);


    save(strcat('./data/condition_',string(timestamp),'.mat'), 'cond1', 'cond2', 'timestamp', 'stateDim_range', 'L', 'inputDim', 'maxiter', 'maxcor', 'gtol', 'k', 'seed');
end

fig = figure;
hold on
plot(stateDim_range, cond1,'-','LineWidth', 1.0);
plot(stateDim_range, cond2, '-','LineWidth', 1.0);
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
legend('$\kappa(\mathcal S)$','$\kappa(\mathcal B\mathcal S)$', 'Interpreter','latex','Location','northeast')
xlabel('state dimension $n$', 'Interpreter','latex') 
ylabel('condition number', 'Interpreter','latex')
%xlim([0,1e+4])
hold off
grid on
grid minor

savefig(fig, strcat('./figures/condition_',string(timestamp),'.fig'));


function [z, flag, iter, resvec, S, Y, rho] = lbfgs(z0, Lam, r, N, L, w, Q, ind, lamb, maxiter, maxcor, reg, gtol, verb, gd)
    if nargin<14 || isempty(verb)
        verb = false;
    end

    if nargin<15 || isempty(gd)
        gd = false;
    end

    % dimenstions of the Toeplitz matrix T
    m = r*L;
    n = N-L+1;

    flag = 1;

    Qw = Q.*w;
    TtQw = transposeFastToeplitz(Qw,Lam,r,N,L);
    zk = z0;
    Tzk = fastToeplitz(zk,Lam,r,N,L);
    QTzk = Q.*Tzk;
    TtQTzk = transposeFastToeplitz(QTzk,Lam,r,N,L);

    lamb_ = zeros(m,1);
    lamb_(ind) = lamb;
    Ttlamb = transposeFastToeplitz(lamb_,Lam,r,N,L);

    % gradient
    gk = TtQTzk - TtQw + reg*zk - Ttlamb;
    
    S = zeros(n,maxiter);
    Y = zeros(n,maxiter);
    rho = zeros(maxiter);

    resvec = zeros(maxiter+1,1);
    
    k = 0;

    % BFGS loop
    while k<=maxiter-1
        k = k+1;

        normgk = norm(gk);
        res = norm([gk; Tzk(ind)-w(ind)]); % residuum
        resvec(k) = res;

        % check convergence
        if normgk <= gtol
            flag = 0;
            break
        end

        % check NaN
        if any(isnan(zk))
            flag = 3;
            break
        end
    
        % new search direction
        % we ommit the negative sign at the search direction
        if gd
            pk = gk; % gradient descent
        else
            pk = inverse_hesse(gk,k,maxiter,maxcor,S,Y,rho); % BFGS direction
        end
        Tpk = fastToeplitz(pk,Lam,r,N,L);
        QTpk = Q.*Tpk;

        % exact line search
        denom = Tpk'*QTpk + reg*(pk'*pk);
        b = (pk'*gk) / denom; % negative sign is ommited as well
        
        % update iterate
        zkk = zk - b*pk; % correct ommited sign
        TtQTpk = transposeFastToeplitz(QTpk,Lam,r,N,L); 
        gkk = gk - b*(TtQTpk + reg*pk);
        
        % update parameters for Hessian
        sk = zkk - zk;
        yk = gkk - gk;
        rhok = sk'*yk;  
        Y(:,k) = yk;
        S(:,k) = sk;
        rho(k) = rhok;

        if verb && mod(k,100)==0
            pause(0.01)
            fprintf('%i\t gk: %e\t res: %e\tb: %e\t denom: %e\n',k ,normgk, res, b, denom);
        end
        
        gk = gkk;
        zk = zkk;
    end

    z = zk;
    iter = k;
    resvec = resvec(1:iter);
    
    if verb
        if normgk <= gtol
            fprintf('BFGS converged after %i iterations, gnorm: %e\tres: %e\n', k, normgk, res)
        else
            fprintf('BFGS not converged after %i iterations, gnorm: %e\tres: %e\n', k, normgk, res)
        end
    end
    return
end


function q = inverse_hesse(z,k,maxiter,maxcor,S,Y,rho)
    q = z; % inverse Hessian approxmiate initialized as identity
    if k==1
        return
    end

    alpha = zeros(maxiter);
    s = max(1,k-maxcor);
    for j = k-1:-1:s
        alpha(j) = (S(:,j)'*q) / rho(j);
        q = q - alpha(j)*Y(:,j);
    end
    gamma = rho(k-1) / (Y(:,k-1)'*Y(:,k-1));
    q = gamma*q;
    for j = s:k-1
        beta = (Y(:,j)'*q) / rho(j);
        q = q + (alpha(j)-beta)*S(:,j);
    end
    return
end