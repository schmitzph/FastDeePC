function Ttz = transposeFastToeplitz(z,Lam,r,N,L)
%TRANSPOSEFASTTOEPLITZ calculates the matrix-vector product T'*z, where the 
%   Toeplitz matrix T is represented via (Lam, r, N, L)

    n = N-L+1;
    y = ifft([reshape(z,[r,L]),zeros(r,N-L)]',N,1); % zero padding to fit dimension N and inverse DFT
    w = Lam'.*y; % attention: Lam' is transposed, conjugate-complex of Lam
    u = sum(w,2);
    v = fft(u,N,1);
    Ttz = v(1:n); % cutoff
    Ttz = real(Ttz);
end