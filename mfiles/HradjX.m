function HradjX = HradjX(X,N,r,p)
% This function computes  H^* applied to X,
% where X is a p*(r+1) by N-r+1 real matrix and r <= N.
% The output is a p by N+1 real matrix.

HradjX = zeros(p,N+1);
for i = 1:N-r+1
  if i < r+2
    HradjX = HradjX + [zeros(p,i-1) reshape(X(1:(r+1)*p-(i-1)*p,i),p,r+2-i) X((r+1)*p-i*p+1:(r+1)*p-(i-1)*p,i+1:N-r+1) zeros(p,i-1)];
  else
    break
  end
end