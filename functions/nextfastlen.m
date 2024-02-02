function [m] = nextfastlen(n, primes)
%NEXTFASTLEN computes the the smallest integer m larger or equal n
%   containing only prime factors from the array primes.

    if (nargin < 2) || isempty(primes)
        m = Inf;
        return
    end
    
    if n == 1
        m = 1;
        return;
    else
        p = primes(1);
        m1 = nextfastlen(fix((n+p-1)/p), primes)*p;
        m2 = nextfastlen(n, primes(2:end));
        m = min(m1,m2);
        return;
    end
end

