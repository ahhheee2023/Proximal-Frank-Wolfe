function Hrx = Hrx(x,r)
% This function computes  H applied to x,
% where x is a p by N+1 real matrix and r <= N.
% The output is an (r+1)*p by N-r+1 real matrix.

p = size(x,1);
N = size(x,2) - 1;

Hrx = zeros((r+1)*p,N-r+1);
for i = 1:N-r+1
  Hrx(:,i) = reshape(x(:,i:i+r),(r+1)*p,1);
end