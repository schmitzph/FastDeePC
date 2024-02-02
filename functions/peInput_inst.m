function u = peInput_inst(inputDim,L,A,B,random,fastlen)
%PEINPUT_INST generates an input signal of dimension inputDim, which is 
%   persistently exciting of order L, for instable systems (A,B) via 
%   stabilizing feedback.

%   The paramter random decides whether the entries of u are drawn randomly
%   or chosen via a specific deterministic rule.

%   If fastlen is True, then the lenght N of u is increased to an
%   optimized signal lenght with respect to FFT

    N = (inputDim+1)*L-1;
    dimState = size(A,1);
    
    % stabilizing feedback via pole placement
    p = 0.9*exp(2i*pi*(1:dimState)/dimState);
    p = 0.5*(conj(circshift(flip(p),-1))+p); % correction of complex conjugacy
    F = place(A,B,p);

    if fastlen
        N = nextfastlen(N,[2,3,5,7]);
    end
    if random
        u = 2*rand(inputDim,N)-1;
    else
        u = zeros(inputDim, N);
        for j = 1:inputDim
            u(j,j*L) = 1;
        end
    end
    
    x = zeros(dimState,1);
    for k = 1:N-1
       u(:,k) = u(:,k) - F*x;
       x = A*x + B*u(:,k);
    end
    return
end


