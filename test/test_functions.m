%%%%%%%%% Test functions

clear
close all
rng(123) % random seed
tol = 1e-10;

addpath('../functions')





k = 1;

%%%%%%%%% spawnSystem
fprintf('\nTest %i: spanwSystem\n', k)
flag(k)=0;

stateDim = 100;
inputDim = 50;
sparsity = 0.5;
specrad = 0.9;

[A, B] = spawnSystem(stateDim, inputDim, sparsity);
[A, B] = spawnSystem(stateDim, inputDim, sparsity, specrad);

ev = eig(A);
maxev = max(abs(ev));
if abs(maxev-specrad) >= tol
    flag(k) = 1;
end

sparsityA = sum(sum(A~=0.0)) / (stateDim^2);
sparsityB = sum(sum(B~=0.0)) / (inputDim*stateDim);

if abs(sparsityA-sparsity)>=tol || abs(sparsityB-sparsity)>=tol
    flag(k) = 2;
end





k = k + 1;

%%%%%%%%% peInput
fprintf('\nTest %i: peInput\n', k)
flag(k) = 0;

L = 50;

U = peInput(inputDim, L, false, false);
T = buildToeplitz(U, L);
if rank(T) < (L*inputDim)
    flag(k) = 1;
end

U = peInput(inputDim, L, true, false);
T = buildToeplitz(U, L);
if rank(T) < (L*inputDim)
    flag(k) = 2;
end

U = peInput(inputDim, L, false, true);
T = buildToeplitz(U, L);
if rank(T) < (L*inputDim)
    flag(k) = 3;
end

U = peInput(inputDim, L, true, true);
T = buildToeplitz(U, L);
if rank(T) < (L*inputDim)
    flag(k) = 4;
end





k = k + 1;

%%%%%%%%% calcState
fprintf('\nTest %i: calcState\n', k)
flag(k) = 0;

X = calcState(U, A, B);

err = norm(norm(X(:,2:end) - A*X(:,1:end-1) - B*U(:,1:end-1)));
if err >= tol
    flag(k) = 1;
end

X = calcState(U, A, B, ones(1,stateDim));
err = norm(norm(X(:,2:end) - A*X(:,1:end-1) - B*U(:,1:end-1)));
if err >= tol
    flag(k) = 2;
end





k = k + 1;

%%%%%%%%% fastToeplitz
fprintf('\nTest %i: fastToeplitz\n', k)
flag(k) = 0;

seq = [X;U];
[r,N] = size(seq);
Lam = representToeplitz(seq, L);
n = N-L+1;

z = rand(n,1)*20 - 10;
Tz = fastToeplitz(z, Lam, r, N, L);
T = buildToeplitz(seq, L);
Tz_ = T*z;

if norm(Tz-Tz_)>=tol
    flag(k) = 1;
end




k = k + 1;

%%%%%%%%% transposeFastToeplitz
fprintf('\nTest %i: transposeFastToeplitz\n', k)
flag(k) = 0;

v = rand(r*L,1)*20 - 10;
Ttv = transposeFastToeplitz(v, Lam, r, N, L);
Ttv_ = T'*v;

if norm(Tz-Tz_)>=tol
    flag(k) = 2;
end




k = k + 1;

%%%%%%%%% z2trajectory
fprintf('\nTest %i: z2trajectory\n', k)
flag(k) = 0;


[X2,U2] = z2trajectory(z, Lam, r, N, L, stateDim);
err2 = norm(norm(X2(:,2:end) - A*X2(:,1:end-1) - B*U2(:,1:end-1)));
if err2 >= tol
    flag(k) = 1;
end





if any(flag)
    fprintf('\nTest failed\n')
    disp(flag)
else
    fprintf('\nTest passed\n')
end
