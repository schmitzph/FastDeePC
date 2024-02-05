function [z, iter, resvec, Hinv] = lbfgs_db(z0, Lam, r, N, L, w, Q, ind, lamb, maxiter, maxcor, reg, gtol, gd)
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

%   gd=true enables gradient descent (no approx. inverse Hessian)

    if nargin<14 || isempty(gd)
        gd = false;
    end

    resvec = zeros(maxiter,1);

    % dimenstions of the Toeplitz matrix T
    m = r*L;
    n = N-L+1;

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
    
    function q = inverse_hesse(z,k) % limited mermory inverse Hessian update
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

    function Hinv = get_inverse_hess(k) % numerical inverse Hessian approximate
        Hinv = eye(n);
        for i = 1:n
            Hinv(:,i) = inverse_hesse(Hinv(:,i),k);
        end
    end
    
    k = 0;

    while k<=maxiter
        k = k+1;

        normgk = norm(gk);
        resk = [gk; Tzk(ind)-w(ind)];
        resvec(k) = norm(resk);
        if normgk <= gtol
            fprintf('\nConverged after %i steps, gnorm: %e\n', k-1, normgk)
            break
        end
    
        % new search direction
        if gd
            pk = gk; % gradient descent, otherwise BFGS
        else
            pk = inverse_hesse(gk,k); % we ommit the negative sign at the search direction
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

        if mod(k,100)==0
            pause(0.01)
            fprintf('\n%i\t gk: %e\tb: %e\t denom: %e\t sk: %e',k, norm(gk),b, denom, norm(sk));
        end
        
        gk = gkk;
        zk = zkk;
    end

    z = zk;
    iter = k;
    resvec = resvec(1:iter);

    if nargout >= 4
        Hinv = get_inverse_hess(k);
        disp('Hessian')
    end
    if normgk > gtol
        fprintf('\nNot converged after %i steps, gnorm: %e\n', k-1, normgk)
    end
    return
end