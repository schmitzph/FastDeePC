clear
close all
rng(123) % random seed

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

% (random) initial conditon x0 and reference stored in w
ker = null([A-eye(stateDim), B]);
w = repmat(ker(:,1), L, 1);
x0 = 2*rand(stateDim,1)-1;
ind = (m-(stateDim+inputDim)+1:m-inputDim);
w(ind) = x0;
Q(ind) = 0.0;

TtQw = transposeFastToeplitz(Q.*w,Lam,r,N,L);
rhs = [TtQw; -x0];
[vec,flag,~,iter_minres,res_minres] = minres(@optSys, rhs, 1e-4,10000,[],[],[], Lam, r, N, L, Q, ind);
z = vec(1:N-L+1);
[c, feas] = cost(z,Lam, r, N, L, w, Q, ind);
fprintf('\nfeasability violation: %e\t cost: %e\n', feas, c);

[X_,U_] = z2trajectory(vec(1:N-L+1), Lam, r, N, L, stateDim);

figure
hold on
plot(X_', 'r')
plot(U_', 'b');


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