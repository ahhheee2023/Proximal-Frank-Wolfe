function [x, iter, fval, cpu_time, history] = proxFW(bx, U, r, sigma, xinit,cres, opts)

% r is the order in system identification
% bx is a m by N+1 matrix
% U is a m*(r+1) by N+1-r matrix

fval_rec = [];
Hxy_res_rec = [];
cst_res_rec = [];

cputime_now = 0;
tstart = tic;

sigmatilde = sqrt(2)*(sigma^2 + norm(bx, 'fro')^2);
Ut = U';
U_hdl = @(X) X*U;
Ut_hdl = @(X) X*Ut;
k = size(U,1);      % k = N+1-r;
m = size(bx, 1);
% q = size(U, 2);
j = r+1;
cst1 = min(j,k);
N = size(bx, 2) - 1;

x = xinit;
Hx = U_hdl( Hrx(x, r) );
% tracenm_Hx = sum(svd(Hx));
% cst_res =  (tracenm_Hx - sigma)/sigma;
y = Hx;
Hxy = Hx - y;
Hxy_res = norm(Hxy, 'fro');

if opts.noise_type == 2
    z = x-bx;
end

iter = 1;
while iter < opts.maxiter
%     Hxy = Hx - y;
%         Hxy_res = norm(Hxy, 'fro');    % using some relative residual?

    if opts.noise_type == 1
        fval = 0.5*norm(x-bx, 'fro')^2;
    elseif opts.noise_type == 2
        fval = sum(sum(abs(z)));
    end
    
    if mod(iter, opts.verbose_freq) == 1
        tracenm_Hx = sum(svd(Hx));
        cst_res =  (tracenm_Hx - sigma)/sigma;
        
        cst_res_rec = [cst_res_rec; cst_res];
        fval_rec = [fval_rec; fval];
        Hxy_res_rec = [Hxy_res_rec; Hxy_res];
        fprintf('Iter = %d: fval = %6.4e, nm(y-Hx) = %6.2e, tracenm(Hx)-sigma = %6.2e \n', iter, fval, Hxy_res, cst_res);
    end
    
    if (Hxy_res < cres && iter > 1) || cputime_now > opts.maxtime  || abs(cst_res) < 1e-6
%         fprintf('Terminate due to the small constraint residual. \n')
        break;
    end
    
    adjHxy = HradjX( Ut_hdl(Hxy), N, r, m );
    beta_t = sqrt(iter);
    %     beta_t = 100*sqrt(iter);
    %     beta_t = 10*sqrt(iter);
    %         beta_t =iter^(0.3);
    if opts.noise_type == 1
        xcirc = x - ( (x - bx) + beta_t*adjHxy ) / (1+beta_t*cst1 );
        nm_xcirc = norm(xcirc, 'fro');
        if nm_xcirc <= sigma
            x = xcirc;
        else
            x = (sigma/nm_xcirc)*xcirc;
        end
    elseif opts.noise_type ==2
        zcirc = z - adjHxy/ cst1;
        sf = abs(zcirc) - 1/(beta_t*cst1);
        sfthreshold = max(sf, 0);
        z = min(sigmatilde/norm(sfthreshold), 1) * ( sign(zcirc).*  sfthreshold);
        x = z + bx;
    end
    
    
    Hx = U_hdl(Hrx(x, r));
    
    yHx = y - Hx;
    [R, s, T] = svds(yHx,1);
    alpha = 2/(iter+2);
    y = (1-alpha)*y - ( (sigma*alpha) *R ) * T';
    
    Hxy = Hx - y;
    Hxy_res = norm(Hxy, 'fro');
    
    if opts.boosting.flag && mod(iter, opts.boosting.freq) == 1
        tracenm_y = sum(svd(y));
        ytilde = y * (sigma/tracenm_y);
        Hxytilde = Hx - ytilde;
        Hxytilde_res = norm(Hxytilde, 'fro');
        
        if Hxy_res > Hxytilde_res
%             keyboard
            fprintf('iter = %d : Boosting succedded.\n', iter);
            y = ytilde;
            Hxy = Hxytilde;
            Hxy_res = Hxytilde_res;
        end
    end
    %% cputime
    iter = iter+1;
    cputime_now = toc(tstart);
    
end
cpu_time = cputime_now;
history.fval_rec = fval_rec;
history.Hxy_res_rec = Hxy_res_rec;
history.cres_orig_rec = cst_res_rec;

