function x = calcState(u,A,B,x0)
%CALCSTATE calculates the state trajectory for the system (A,B) 
%   with respect to the input signal u and the initial condtion x0.
%   If no initial condition x0 is provided the state is randomly
%   intialized.

    stateDim = size(A,1);
    if nargin < 4 || isempty(x0)
        x0 = 2*rand(stateDim,1)-1;
    end
    N = size(u,2);
    x = zeros(stateDim, N);
    x(:,1) = x0;
    for j = 2:N
        x(:,j) = A*x(:,j-1) +  B*u(:,j-1);
    end
    return
end