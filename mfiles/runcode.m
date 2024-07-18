% test the proxFW method for the system identification problems
clear;

rand('seed', 2012);
randn('seed', 2011);

N = 3000; rbar = 20; m = 20;
p = 5;
% opts.noise_type = 1;  % Gaussian noise
opts.noise_type = 2;   % Laplacian noise
[U, bx, r, sigma_apprx] =  generate(N, rbar, m, p, opts.noise_type);


fprintf('Generations done.\n')

xinit = zeros(m, N+1);  % Then yinit = Hrx(xinit)*U = zeros(m(r+1), N+1-r) is feasible
opts.maxiter = 2000;
% maxiter = 3000;
opts.verbose_freq = 100;
opts.maxtime = 3600;
boosting.flag = 0;
boosting.freq = 10;
opts.boosting = boosting;
Hxy_tol = 1e-8;

fprintf('Start of the algorithm.\n')
fprintf('Maximum iterations number is set as %d\n', opts.maxiter);
[x1, iter,  fval1, cpu_time1, history1] = proxFW(bx, U, r, sigma_apprx, xinit, Hxy_tol, opts);

% fig1 = figure;
% plot(1:length(history.fval_rec), history.fval_rec, 'b-.', 'LineWidth', 2);
% axis([0, length(history.fval_rec), 0, max(history.fval_rec)])
% xlabel('iteration/verbose_{freq}', 'Interpreter', 'tex');
% ylabel('function value', 'Interpreter', 'tex')
% 
% % fig2 = figure;
% % semilogy(1:length(history.cres_rec)-1, history.cres_rec(2:length(history.fval_rec)), 'b-.', 'LineWidth', 1);
% % axis([0, length(history.cres_rec), 1e-5,1e4])
% % xlabel('iteration');
% % ylabel('log(||y-H(x)||)', 'Interpreter', 'tex')
% 
% fig3 = figure;
% cres_orig = history.cres_orig_rec;
% plot(1:length(cres_orig), cres_orig, 'b-', 'LineWidth', 2);
% xlabel('iteration/verbose_{freq}', 'Interpreter', 'tex');
% ylabel('||H(x) - \sigma|| / \sigma', 'Interpreter', 'tex')






