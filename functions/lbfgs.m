function [z, flag, iter, resvec, Hinv] = lbfgs(z0, Lam, r, N, L, w, Q, ind, lamb, maxiter, maxcor, reg, gtol, verb, gd)
%LBFGS limited-memory BFGS

%   z0 intial guess for solution

%   (Lam, r, N, L) describe the Toeplitz matrix

%   w contains the initial condition and the reference

%   Q diagonal weight matrix in the OCP

%   ind indeces where the initial condition is active

%   lamb Lagrangian multiplier

%   maxiter maximal number of iterations in each lBFGS run

%   maxcor maximal past steps used in inverse Hessian approximation
%       during lBFGS

%   gtol tolerance for the gradient

%   reg regularization parameter in OCP

%   z solution

%   flag=0 convergence, flag=1 not converged

%   sequence of all residual norms

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
    
    % limited mermory inverse Hessian update
    function q = inverse_hesse(z,k)
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

    % returns numerical inverse Hessian approximate
    function Hinv = get_inverse_hess(k)
        Hinv = eye(n);
        for i = 1:n
            Hinv(:,i) = inverse_hesse(Hinv(:,i),k);
        end
    end
    
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
    
        % new search direction
        % we ommit the negative sign at the search direction
        if gd
            pk = gk; % gradient descent
        else
            pk = inverse_hesse(gk,k); % BFGS direction
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

    if nargout >= 7
        Hinv = get_inverse_hess(k);
    end
    
    if verb
        if normgk <= gtol
            fprintf('BFGS converged after %i iterations, gnorm: %e\tres: %e\n', k, normgk, res)
        else
            fprintf('BFGS not converged after %i iterations, gnorm: %e\tres: %e\n', k, normgk, res)
        end
    end
    return
end