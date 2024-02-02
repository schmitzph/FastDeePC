function [X,U] = z2trajectory(z, Lam, r, N, L, stateDim)
%Z2TRAJECTORY calculates a system trajectory (X,U) from a column 
%   selector z.

% The Toeplitz matrix is represented by (Lam, r, N, L)
    Tz = fastToeplitz(z, Lam, r, N, L);
    W = flip(reshape(Tz,r,[]),2); % reshape and permute
    X = W(1:stateDim,:);
    U = W(stateDim+1:end,:);
    return
end