function Lam = representToeplitz(seq, L)
%REPRESENTTOEPLITZ returns the Lambda matrix for factorization of
% a Toeplitz matrix of depth L represented by the sequence seq
    N = size(seq,2);
    Lam = fft(circshift(seq,-L+1,2), N, 2);
end