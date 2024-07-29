function [x, y, iter, history] =  FW_lgl(A, b, c1, K, Grps, opts)
% % This solves
%  % min ||Ax-b||_1
% %  s.t. Omega(x) <= c1
% % where Omega(.) is the latent group norm

% % Inputs
% Grps: a cell containint the index sets of the groups

fval_rec = zeros(1,1);
FWgap_rec = zeros(1,1);

% m = size(A,1);
n = size(A,2);

if isfield(opts, 'noiseLev')
    c2 = opts.noiseLev;
else
    lambdaA = svds(A,1);
    c2 = lambdaA*c1+norm(b);   % the upper bound for y=Ax-b
end

if isfield(opts, 'xinit')
    xinit = opts.xinit;
else
    xinit = zeros(n, 1);
end

if isfield(opts, 'maxiter')
    maxiter = opts.maxiter;
else
    maxiter = inf;
end

if isfield(opts, 'gap_tol')
    gap_tol = opts.gap_tol;
else
    gap_tol = 1e-8;
end

if isfield(opts, 'verbose_freq')
    verbose_freq = opts.verbose_freq;
else
   verbose_freq = 1e-8;
end

x = xinit;
% y = A*xinit - b;

iter = 1;
fprintf('%6s    %8s   %8s \n', 'iter', 'fval', 'FW_gap');
while 1

    beta = sqrt(iter);
    Axb = A*x - b;
    fval= norm(Axb, 1);
    fval_rec(iter,1) = fval;
    
    % % update of y
    yplus = Axb - 1/beta;
    ytilde = sign(yplus).*max(yplus, 0);   % soft thresholding
    y = min(c2/norm(ytilde), 1) * ytilde;
    
    % % update of x using the FW linear oracle
    z = beta*(A'*(Axb-y));
    norms = zeros(K,1);
    for k=1:K
        norms(k,1) = norm(z(Grps{k}));
    end

    [normmax, kmax] = max(norms);
    % u = zeros(n,1);
    %     u(G{kmax}) = (-c1/normmax)*z(Grps{kmax});
    sk = (c1/normmax)*z(Grps{kmax});
    dneg= sk  +  x(Grps{kmax});
    alpha = 2/(iter+2);
    x(Grps{kmax}) = x(Grps{kmax}) - alpha*dneg;
    
    FWgap = sk'*dneg;
    FWgap_rec(iter, 1) = FWgap;
    
    % % display
    if mod(iter, verbose_freq) == 1
        fprintf('%6d:   %8.2f    %8.2f\n', iter, fval, FWgap);
    end
    
    % % check Termination
    if iter >= maxiter
        fprintf('Maximum iteration numbers used.\n')
        break;
    end
    if FWgap < gap_tol
        fprintf('The Frank-Wolfe gap is small enough.\n');
        break;
    end
    %     if abs( (fval_rec(itr) - fval_rec(itr-1)) / fval_rec(itr) ) < 1e-8
    %         fprintf('The successive changes of the funciton value is small enough. \n')
    %         break;
    %     end
    
    iter = iter +1;
end

history.fval_rec = fval_rec;
history.FWgap_rec = FWgap_rec;

