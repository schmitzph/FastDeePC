function T = buildToeplitz(seq,L)
% BUILDTOEPLITZ returns the numerical Toeplitz matrix.

    [r,N] = size(seq);
    n = N-L+1;
    m = L*r;
    T = zeros(m,n);
    for i = 1:L
        T((L-i)*r+1:(L-i+1)*r, :) = seq(:, i:n+i-1);
    end
end