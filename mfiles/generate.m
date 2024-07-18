% Generates problem for system identification
%
% x_{t+1}  = A*x_t + B*u_t
% bx_t     = C*x_t + D*u_t
%
% where A is r by r, B is r by p, C is m by r, D is m by p
% the input u is p by N+1 and the output bx is m by N+1
%

function [U, bx, r, sigma_apprx] = generate(N, rbar, m, p, noise_type)
%   N = 1299; rbar = 10; m = 10;
%   p = 5;
% noise_type = 1: Gaussian noise
% noise_type =2: Laplacian noise

A = randn(rbar,rbar);
A = A/norm(A); % normalize A
B = randn(rbar,p);
B = B/norm(B); % normalize B
C = randn(m,rbar);
C = C/norm(C);
D = randn(m,p);
D = D/norm(D);
u = randn(p,N+1);

x = randn(rbar,1); % randomly generate state variable

bx = zeros(m,N+1); % collection of all outputs

for i = 1:N+1
    xnew = A*x + B*u(:,i);
    bx(:,i) = C*x + D*u(:,i);
    x = xnew;
end

if noise_type == 1
    bx = bx + 5e-2*randn(m,N+1); % add Gaussian noise
elseif noise_type == 2
    bx = bx + 5e-2*(randn(m,N+1).*randn(m,N+1)-rand(m,N+1).*randn(m,N+1));  % add Laplacian noise
end

% the rbar above are real system order. the r below is the number of blocks
% required
r = 2*rbar + 1;

U = Hrx(u,r);
U = null(U);
q = size(U,2);

%   save sysP8 m N p r U bx

Hrbx = Hrx(bx, r)*U;
% sigma_apprx = sqrt( m*sum( sum(Hrbx.*Hrbx) ) );
sigma_apprx = sum(svd(Hrbx))*0.95;