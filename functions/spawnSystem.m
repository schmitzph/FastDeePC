function [A,B] = spawnSystem(stateDim, inputDim, sparsity, specrad)
%SPAWNSYSTEM generates a system (A,B). The matrices A and B are drawn
%   randomly such that they are sparse with given sparsity and A has
%   spectral radius specrad.

    if (nargin < 4) || isempty(specrad)
        specrad = 0.9;
    end

    % random dense matrix
    A = 2*rand(stateDim)-1;

    % apply sparsity
    indA = randsample(stateDim^2, floor(sparsity*stateDim^2));
    A = reshape(A,1,[]);
    A(indA) = 0.0;
    A = reshape(A,stateDim,stateDim);

    % adjust spectral radius
    ev = eig(A);
    maxev = max(abs(ev));
    A = specrad * A / maxev;
    
    % random dense matrix
    B = 2*rand(stateDim, inputDim)-1;

    % apply sparsity
    indB = randsample(stateDim*inputDim, floor(sparsity*stateDim*inputDim));
    B = reshape(B,1,[]);
    B(indB) = 0.0;
    B = reshape(B,stateDim,inputDim);
    
    % check controllability
    if rank(ctrb(A,B)) < stateDim
        disp('System not controllable!')
    end
    
    return
end