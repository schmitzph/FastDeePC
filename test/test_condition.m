%%%%%%%%% Test preconditioning of augmented Lagrangian limited-memory BFGS

clear
close all
seed = 123;
rng(seed) % random seed

timestamp = datetime('now');
timestamp.Format = 'yyyy-MM-dd_HHmmss';

addpath('../functions')

% system dimensions
stateDim_range = [10,20,30,40,50,60,70,80,100,120,140,160,180,200,220,240,260,280,300,350,400,450,500];
inputDim = 50;

% prediction horizon
L = 50;

% BFGS parameters
maxiter = 1000;
maxcor = 1000;
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
    
    % (random) initial conditon x0 and reference stored in w
    ker = null([A-eye(stateDim), B]);
    w = repmat(ker(:,1), L, 1);
    x0 = 2*rand(stateDim,1)-1;
    ind = (m-(stateDim+inputDim)+1:m-inputDim);
    w(ind) = x0;
    
    z0 = 2*rand(N-L+1,1)-1;
    lamb0 = rand(length(ind),1);

    [~, flag, ~, ~, Hinv] = lbfgs(z0,Lam,r,N,L,w,Q,ind,lamb0,maxiter,maxcor,reg,gtol,verb,false);

    % compute condition numbers
    T = buildToeplitz(seq, L);
    TtQT = T'*(Q.*T);
    s1 = svd(TtQT);
    cond1(k) = s1(1)/s1(L*inputDim+stateDim);

    s2 = svd(Hinv*TtQT);
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