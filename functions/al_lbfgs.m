function [z, flag, iter, resvec] = al_lbfgs(Lam, r, N, L, w, Q, ind, maxiter, maxcor, reg, gtol, ftol, murange, z0, lamb0, verb, gd)
%AL_LBFGS augmented Lagrangian limited-memory BFGS

%   (Lam, r, N, L) describe the Toeplitz matrix

%   w contains the initial condition and the reference

%   Q diagonal weight matrix in the OCP

%   ind indeces where the initial condition is active

%   maxiter maximal number of iterations in each lBFGS run

%   maxcor maximal past steps used in inverse Hessian approximation
%       during lBFGS

%   reg regularization parameter in OCP

%   gtol tolerance for the gradient

%   murange range of the penalty paramter

%   z0, lamb0 initial guess for the solution and dual variable

%   verb=true verbose

%   noHess=true runs gradient descent instead of BFGS

%   z solution

%   flag=0 convergence, flag=1 inner BFGS did not converged, flag=2
%       feasibility violation

%   iter number of iteratrions

    if verb
        disp('This is aL l-BFGS')
    end

    if nargin<15 || isempty(lamb0)
        lamb = rand(length(ind),1);
    else
        lamb = lamb0;
    end
    if nargin<14 || isempty(z0)
        z = 2*rand(N-L+1,1)-1;
    else
        z = z0;
    end
    if nargin<16 || isempty(verb)
        verb=false;
    end
    if nargin<17 || isempty(gd)
        gd=false;
    end

    resvec = zeros(maxiter*length(murange));

    iter = 0;
    
    for j = 1:length(murange)
        mu = murange(j);
        Q(ind) = Q(ind)+mu; % including penelty in weight
        [z, flag, k, resvec_] = lbfgs(z,Lam,r,N,L,w,Q,ind,lamb,maxiter,maxcor,reg,gtol,verb,gd);
        resvec(iter+1:iter+k) = resvec_;
        v = fastToeplitz(z,Lam,r,N,L); % evaluate solution z
        err = v(ind)-w(ind); % deviation from the OCP initial condition
        nerr = norm(err);
        lamb = lamb - mu*err; % update Lagragian multiplier
        iter = iter + k;
        if verb
            fprintf('feas. viol.: %e\n\n',nerr);
        end
        if nerr <= ftol
            break
        else
            flag = 2;
        end
    end

    resvec = resvec(1:iter);
    if verb
        if flag == 0
            fprintf('aL converged after %i iterations, res: %e\t feas. viol.: %e\n',iter, resvec(end), nerr);
        else
            fprintf('aL not converged (flag %i) after %i iterations, res: %e\t feas. viol.: %e\n',flag, iter, resvec(end), nerr);
        end
    end
end

