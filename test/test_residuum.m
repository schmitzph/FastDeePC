%%%%%%%%% Test augmented Lagrangian limited-memory BFGS vs
%%%%%%%%%      augmented Lagrangian gradient descent vs
%%%%%%%%%      MINRES
%%%%%%%%% with respect to their residua

clear
close all
seed = 123;
rng(seed) % random seed

timestamp = datetime('now');
timestamp.Format = 'yyyy-MM-dd_HHmmss';

addpath('../functions')

if ~exist('./data', 'dir')
    mkdir('./data');
end

if ~exist('./figures', 'dir')
    mkdir('./figures');
end


% LTI system (A,B)
stateDim = 100;
inputDim = 50;
[A, B] = spawnSystem(stateDim, inputDim, 0.5, 0.9);

% prediction horizon
L = 50;



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

Q = ones(m,1); % OCP weights
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
ftol = 1e-8;
verb = true;
murange = linspace(1.0e1,1.0e+4,10);

%%%%%%%%%%% augmented Lagrangian l-BFGS with fast matrix-vector multiplication
fprintf('\n aL l-BFGS (fast matrix-vector) \n')

[z_bfgs, flag_bfgs, iter_bfsgs, resvec_bfgs] = al_lbfgs(Lam, r, N, L, w, Q, ind, maxiter, maxcor, reg, gtol, ftol, murange, z0, lamb0, verb, false);

%%%%%%%%%%% augmented Lagrangian gradient descent with fast matrix-vector multiplication
fprintf('\n aL GD (fast matrix-vector) \n')

[z_gd, flag_gd, iter_gd, resvec_gd] = al_lbfgs(Lam, r, N, L, w, Q, ind, maxiter, maxcor, reg, gtol, ftol, murange, z0, lamb0, verb, true);


%%%%%%%%%%% MINRES %%%%%%%%%
fprintf('\nMINRES fast Toeplitz matrix-vector multiplication\n')

TtQw = transposeFastToeplitz(Q.*w,Lam,r,N,L);
rhs = [TtQw; -x0];
[vec,flag_minres,~,iter_minres,resvec_minres] = minres(@optSys, rhs, 5e-10,10000,[],[],[], Lam, r, N, L, Q, ind);





save(strcat('./data/residuum_',string(timestamp),'.mat'), 'timestamp', 'stateDim', 'inputDim', 'L', 'maxiter', 'maxcor', 'gtol', 'ftol', 'murange', 'seed')

fig = figure;
hold on
plot(resvec_bfgs,'-','LineWidth', 1.0);
plot(resvec_minres, '-','LineWidth', 1.0);
plot(resvec_gd, '-','LineWidth', 1.0);
%set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
legend('aL l-BFGS','MINRES', 'aL GD', 'Interpreter','latex','Location','northeast')
xlabel('iteration', 'Interpreter','latex') 
ylabel('residual norm', 'Interpreter','latex')
xlim([0,1e+4])
hold off
grid on
grid minor

savefig(fig, strcat('./figures/residuum_',string(timestamp),'.fig'));


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
