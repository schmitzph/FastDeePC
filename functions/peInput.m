function u = peInput(inputDim, L, random, fastlen)
%PEINPUT generates an input signal of dimension inputDim, which is 
%   persistently exciting of order L.

%   The paramter random decides whether the entries of u are drawn randomly
%   or chosen via a specific deterministic rule.

%   If fastlen is True, then the lenght N of u is increased to an
%   optimized signal lenght with respect to FFT

    N = (inputDim+1)*L-1;
    if fastlen
        N = nextfastlen(N,[2,3,5,7]);
    end
    if random
        u = rand(inputDim,N);
        return
    end
    u = zeros(inputDim, N);
    for j = 1:inputDim
        u(j,j*L) = 1;
    end
    return
end


