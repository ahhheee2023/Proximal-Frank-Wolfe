function x = sft_bc( y, beta, sigma)
% this computes the prox of L1 norm constrained in a ball:
% min ||x||_1 + beta||x-y||^2/2
% s.t.  ||x|| <= sigma

ybeta = abs(y) - 1/beta;
ybeta_thr = max(ybeta,0);
x = min( sigma/norm(ybeta_thr), 1) * ( ( ybeta_thr.*sign(y) ) );
