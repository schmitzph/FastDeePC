clear
close all
rng(123) % random seed

timestamp = datetime('now');
timestamp.Format = 'yyyy-MM-dd_HHmmss';

addpath('../functions')

% system dimensions
stateDim_range = [10,20,30,40,50];%,60,70,80,100,120,140,160,180,200,220,240,260,280,300,350,400,450,500,600,700,800,900,1000];

tup = [[50;50], [60;60]]; % inputDimesnion and prediction horizon L

% aL BFGS Parameters
maxiter = 1000;
maxcor = 1000;
gtol = 1e-5;
ftol = 1e-7;
verb = true;

murange = linspace(1.0e1,1.0e+4,10);


% results(:,1) <- L
% results(:,2) <- inputDim
% results(:,3) <- stateDim
% results(:,4) <- execution time
% results(:,5) <- total iterations
% results(:,6) <- flag
% results(:,7) <- residual norm

results = [];

for k1 = 1:size(tup,2)
    for k2 = 1:length(stateDim_range)
        inputDim = tup(1,k1);
        L = tup(2,k1);
        stateDim = stateDim_range(k2);

        fprintf('\n\n\n\n########## New round #######\nstatedim: %i\tinputDim: %i\t L: %i\n', stateDim, inputDim, L);

        % generate system (A,B)
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
        Q(ind) = 0.0;

        % relative tolerances
        gtolr = gtol * sqrt(n);
        ftolr = ftol * sqrt(stateDim);

        fprintf('gtolr: %e\t ftolr: %e\n', gtolr, ftolr);

        tic;
        [z, flag, iter, resvec] = al_lbfgs(Lam, r, N, L, w, Q, ind, maxiter, maxcor, reg, gtolr, ftolr, murange, [], [], verb, false);
        t = toc;
        fprintf('Elapsed time is %f seconds.\n', t)

        results(end+1, 1) = L;
        results(end, 2) = inputDim;
        results(end, 3) = stateDim;
        results(end, 4) = t;
        results(end, 5) = iter;
        results(end, 6) = flag;
        results(end, 7) = resvec(iter);

        save(strcat('./data/exectime_',string(timestamp),'.mat'));
    end
end

% create figures

% stateDim
data1 = results(1:length(stateDim_range),3);
data2 = results(length(stateDim_range)+1:2*length(stateDim_range),3);
% exectime
data3 = results(1:length(stateDim_range),4);
data4 = results(length(stateDim_range)+1:2*length(stateDim_range),4);
% iterations
data5 = results(1:length(stateDim_range),5);
data6 = results(length(stateDim_range)+1:2*length(stateDim_range),5);
% mean time
data7 = data3./data5;
data8 = data4./data6;
% asymptote
data9 = data2.^2 .* log(data2);
data9 = data9 / data9(1) * data8(1); % correction

fig1 = figure;
hold on
plot(data1,data3,'-','LineWidth', 1.0);
plot(data2,data4,'-','LineWidth', 1.0);
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
legend('$m=L=50$','$m=L=100$', 'Interpreter','latex','Location','northeast')
xlabel('state dimension $n$', 'Interpreter','latex') 
ylabel('total execution time [s]', 'Interpreter','latex')
%xlim([0,1e+4])
hold off
grid on
grid minor
savefig(fig, strcat('./figures/exectime_',string(timestamp),'.fig'));


fig2 = figure;
hold on
plot(data1,data5,'-','LineWidth', 1.0);
plot(data2,data6,'-','LineWidth', 1.0);
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
legend('$m=L=50$','$m=L=100$', 'Interpreter','latex','Location','northeast')
xlabel('state dimension $n$', 'Interpreter','latex') 
ylabel('total number of BFGS iterations', 'Interpreter','latex')
%xlim([0,1e+4])
hold off
grid on
savefig(fig, strcat('./figures/iterations_',string(timestamp),'.fig'));


fig3 = figure;
hold on
plot(data1,data7,'-','LineWidth', 1.0);
plot(data2,data8,'-','LineWidth', 1.0);
plot(data2,data9,'-','LineWidth', 1.0);
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
legend('$m=L=50$','$m=L=100$', '$\mathcal O(n^2\log n)$','Interpreter','latex','Location','northeast')
xlabel('state dimension $n$', 'Interpreter','latex') 
ylabel('mean execution time per BFGS iteration [s]', 'Interpreter','latex')
%xlim([0,1e+4])
hold off
grid on
savefig(fig, strcat('./figures/meantime_',string(timestamp),'.fig'));
