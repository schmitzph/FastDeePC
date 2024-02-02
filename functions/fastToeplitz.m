function Tz = fastToeplitz(z, Lam, r, N, L)
%FASTTOEPLITZ calculates the matrix-vector product T*z, where the 
%   Toeplitz matrix T is represented via (Lam, r, N, L)

    y = ifft([z;zeros(L-1,1)],N,1); % zero padding to fit dimension N and inverse DFT
    v = fft(transpose(Lam).*y,N,1);
    Tz = reshape(transpose(v(1:L,:)), [r*L,1]); % cutoff and reshape into tall L*r vector
    Tz = real(Tz); % eliminate numerical effects
end