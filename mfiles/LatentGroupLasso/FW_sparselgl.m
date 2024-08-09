function [x, y, iter, history] =  FW_sparselgl(f_hdl, f_args, n, mu, c1, K, Grps, opts)
% % This solves
%  % min f(x) + mu*||x||_1
% %  s.t. Omega(x) <= c1
% % where Omega(.) is the latent group norm

% % Inputs
% Grps: a cell containint the index sets of the groups
% fun_hdl: the function handle computes the function value and gradien of f(x)
% f_args: the parameters used in the fun_hdl

fval_rec = zeros(1,1);
FWgap_rec = zeros(1,1);
lc_res_rec = zeros(1,1);


if isfield(opts, 'L_f')
    L_f = opts.L_f;
else
    error('Need to set opts.L_f : upper bound of the Lipschitz constant of grad(f) \n')
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
% 
% if isfield(opts, 'gap_tol')
%     gap_tol = opts.gap_tol;
% else
%     gap_tol = 1e-8;
% end

if isfield(opts, 'scale_beta')
    scale_beta = opts.scale_beta;
else
    scale_beta = 1;
end

if isfield(opts, 'verbose_freq')
    verbose_freq = opts.verbose_freq;
else
   verbose_freq = 1e-8;
end

x = xinit;
y = x;
grad = f_hdl(x, f_args);

iter = 1;
fprintf('%6s    %8s   %8s   %8s    %8s \n', 'iter', 'fval', 'fval_sc', 'lc_res',  'FW_gap');
while 1
    beta = sqrt(iter)*scale_beta;
    
    % % update of x
    v_tmp = L_f*x + beta*y -grad;   % the right formula
%     v_tmp = L_f*y+beta*x - grad;    % these parameter L_f and beta made a better recovery..
    xplus= abs(v_tmp) - mu;   
    xtilde = (sign(v_tmp).*max(xplus, 0)) / (beta+L_f);
    nm_xtld = norm(xtilde);
    if nm_xtld < c1
        x = xtilde;
    else
        x = (c1/xtilde)*xtilde;
    end
        
    % % update of y using the FW linear oracle
    z = beta*(y-x);
    norms = zeros(K,1);
    for k=1:K
        norms(k,1) = norm(z(Grps{k}));
    end

    [normmax, kmax] = max(norms);
    zGkmax = z(Grps{kmax});
    sk = (c1/normmax)* zGkmax;
    dneg= sk  +  y(Grps{kmax});
    alpha = 2/(iter+2);
    y(Grps{kmax}) = y(Grps{kmax}) - alpha*dneg;

    
    [grad, fval] = f_hdl(x,f_args);   % update the gradient and function value
    fval_rec(iter,1) = fval + mu*norm(x,1);
    
    lc_res = norm(x - y);
    lc_res_rec(iter,1) =lc_res;
    
    FWgap = zGkmax'*dneg;
    FWgap_rec(iter, 1) = FWgap;
    
    % % display
    if  iter > 1 && mod(iter, verbose_freq) == 0
        fprintf('%6d:   %8.4e    %8.2e   %8.2e   %8.2e\n', iter, fval, fval_rec(iter,1) - fval_rec(iter-1,1), lc_res, FWgap);
    end
    
    % % check Termination
    if iter >= maxiter
        fprintf('Maximum iteration numbers used.\n')
        break;
    end
%     if FWgap < gap_tol
%         fprintf('The Frank-Wolfe gap is small enough.\n');
%         break;
%     end
        if iter > 1 && abs( (fval_rec(iter) - fval_rec(iter-1)) / fval_rec(iter) ) < 1e-12
            fprintf('The successive changes of the funciton value is small enough. \n')
            break;
        end
    
    iter = iter +1;
end

history.fval_rec = fval_rec;
history.FWgap_rec = FWgap_rec;
history.lc_res_rec = lc_res_rec;

