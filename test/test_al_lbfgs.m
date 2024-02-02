%%%%%%%%% Test augmented Lagrangian limited-memory BFGS

clear
close all
rng(123) % random seed
gtol = 1e-5;

addpath('../functions')


% LTI system (A,B)
stateDim = 50;
inputDim = 20;
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

Q = ones(m,1); % [Qx,Qu]
reg = 0.0; % regularization parameter

% (random) initial conditon w0 and reference stored in w
ker = null([A-eye(stateDim), B]);
w = repmat(ker(:,1), L-1, 1);
w0 = repmat(ker(:,2), 2, 1);
ind = (m-2*r+1:m);
w(ind) = w0;

% al lBFGS parameters
maxiter = 1000;
maxcor = 1000;
gtol = 1e-5;
murange = linspace(1.0e1,1.0e+4,10);
z = [];
lamb = [];
debug = true;


z = al_lbfgs(Lam, r, N, L, w, Q, ind, maxiter, maxcor, reg, gtol, murange, z, lamb, debug);

% show trajectory
[X_,U_] = z2trajectory(z, Lam, r, N, L, stateDim);

figure
hold on
plot(X_', 'r')
plot(U_', 'b');

