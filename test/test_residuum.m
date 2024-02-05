%%%%%%%%% Test augmented Lagrangian limited-memory BFGS vs
%%%%%%%%%      augmented Lagrangian gradient descent vs
%%%%%%%%%      MINRES
%%%%%%%%% with respect to their residua

clear
close all
rng(123) % random seed

addpath('../functions')


% LTI system (A,B)
stateDim = 40;
inputDim = 20;
[A, B] = spawnSystem(stateDim, inputDim, 0.5, 0.9);

% prediction horizon
L = 20;



% random pe input signal with FFT-optimized length; pe order L+stateDim
U = peInput(inputDim, L+stateDim, true, true);
% corresponding state sequence
X = calcState(U, A, B);

% Toeplitz matrix of size (m x n) with depth L represented by the data seq and Lam,
% respectively
seq = [X;U];
[r,N] = size(seq);
Lam = fft(circshift(seq,-L+1,2), N, 2);
n = N-L+1;
m = r*L;

% optimal control problem
%   min_(x,u)     sum_k ||xk-wx||_Qx^2 + ||uk-wu||_Qu^2 + reg*||uk||^2
%   s.t. xkk = Axk + Buk,     x0 = w0

Q = ones(m,1); % [Qx,Qu]
%Q(m-r+1:m-r+stateDim) = 0.0;
reg = 0.0; % regularization parameter

% (random) initial conditon x0 and reference stored in w
ker = null([A-eye(stateDim), B]);
w = repmat(ker(:,1), L, 1);
x0 = 2*rand(stateDim,1)-1;
ind = (m-(stateDim+inputDim)+1:m-inputDim);
w(ind) = x0;
Q(ind) = 0.0;


z0 = 2*rand(N-L+1,1)-1;
lamb0 = rand(length(ind),1);

maxiter = 1000;
maxcor = 1000;
gtol = 1e-6;
murange = linspace(1.0e1,1.0e+4,10);

%%%%%%%%%%% augmented Lagrangian l-BFGS with fast matrix-vector multiplication
fprintf('\n aL l-BFGS (fast matrix-vector) \n')

z = z0;
lamb = lamb0;
Q_ = Q;

iter_bfgs = 0; % iteration counter
res_bfgs = [];

for j = 1:length(murange)
    mu = murange(j);
    Q_(ind) = mu; % including penelty in weight
    [z, iter, resvec, Hinv] = lbfgs_db(z,Lam,r,N,L,w,Q_,ind,lamb,maxiter,maxcor,reg,gtol);
    v = fastToeplitz(z,Lam,r,N,L); % evaluate solution z

    % calculate residuum
    res_bfgs = [res_bfgs; resvec];

    % update Lagrangian multiplier
    err = v(ind)-x0;
    lamb = lamb - mu*err;
    fprintf('\nfeasability violation: %e\n',norm(err));

    % increment iteration counter
    iter_bfgs = iter_bfgs + iter;
end

%%%%%%%%%%% augmented Lagrangian gradient descent with fast matrix-vector multiplication
fprintf('\n aL GD (fast matrix-vector) \n')
z = z0;
lamb = lamb0;
Q_ = Q;

iter_gd = 0; % iteration counter
res_gd = [];

for j = 1:length(murange)
    mu = murange(j);
    Q_(ind) = mu; % including penelty in weight
    [z, iter, resvec] = lbfgs_db(z,Lam,r,N,L,w,Q_,ind,lamb,maxiter,maxcor,reg,gtol,true);
    v = fastToeplitz(z,Lam,r,N,L); % evaluate solution z

    % calculate residuum
    res_gd = [res_gd; resvec];

    % update Lagrangian multiplier
    err = v(ind)-x0;
    lamb = lamb - mu*err;
    fprintf('\nfeasability violation: %e\n',norm(err));

    % increment iteration counter
    iter_gd = iter_gd + iter;
end


%%%%%%%%%%% MINRES %%%%%%%%%
fprintf('\nMINRES fast Toeplitz matrix-vector multiplication\n')

TtQw = transposeFastToeplitz(Q.*w,Lam,r,N,L);
rhs = [TtQw; -x0];
[vec,flag,~,iter_minres,res_minres] = minres(@optSys, rhs, 5e-10,10000,[],[],[], Lam, r, N, L, Q, ind);


fig = figure;
hold on
plot(res_bfgs,'LineWidth', 1.5);
plot(res_minres, '-.','LineWidth', 1.5);
plot(res_gd, '--','LineWidth', 1.5);
yline(gtol, ':');
set(gca, 'YScale', 'log')
legend('aL l-BFGS','MINRES', 'aL GD', 'Interpreter','latex','Location','northeast')
xlabel('iteration', 'Interpreter','latex') 
ylabel('residuum', 'Interpreter','latex')
%xlim([0,4e+3])
%ylim([1e+0,4e+4])
hold off
grid on
grid minor


function fval = optSys(vec, Lam, r, N, L, Q, ind)
    m = r*L;
    n = N-L+1;
    z = vec(1:n);
    lamb = vec(n+1:end);
    lamb_ = zeros(m,1);
    lamb_(ind) = lamb;
    Tz = fastToeplitz(z,Lam,r,N,L);
    QTz = Q.*Tz;
    TtQTz = transposeFastToeplitz(QTz,Lam,r,N,L);
    Ttlamb = transposeFastToeplitz(lamb_,Lam,r,N,L);
    fval = [TtQTz - Ttlamb; -Tz(ind)];
    return
end
