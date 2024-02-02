function z = al_lbfgs(Lam, r, N, L, w, Q, ind, maxiter, maxcor, reg, gtol, murange, z, lamb, debug)
%AL_LBFGS augmented Lagrangian limited-memory BFGS

%   (Lam, r, N, L) describe the Toeplitz matrix

%   w contains the initial condition and the reference

%   Q diagonal weight matrix in the OCP

%   ind indeces where the initial condition is active

%   maxiter maximal number of iterations in each lBFGS run

%   maxcor maximal past steps used in inverse Hessian approximation
%       during lBFGS
%   reg regularization parameter in OCP

%   murange range of the penalty paramter

%   z, lamb initial guess for the solution and dual variable

    if nargin<14 || isempty(lamb)
        lamb = rand(2*r,1);
    end
    if nargin<13 || isempty(z)
        z = 2*rand(N-L+1,1)-1;
    end
    
    for j = 1:length(murange)
        mu = murange(j);
        Q(ind) = Q(ind)+mu; % including penelty in weight
        z = lbfgs(z,Lam,r,N,L,w,Q,ind,lamb,maxiter,maxcor,reg,gtol);
        v = fastToeplitz(z,Lam,r,N,L); % evaluate solution x
        err = v(ind)-w(ind); % deviation from the OCP initial condition
        fprintf('\nfeasability violation: %e\n',norm(err));
        lamb = lamb - mu*err; % update Lagragian multiplier
    end
end

