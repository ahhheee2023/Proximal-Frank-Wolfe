%  The generations are as in the way decribed in Section~2 and Section~5.1 of the reference
%  "[1] Sparse overlapping set Lasso for multitask learning and its Application to fMRI analysis"
%  Alos ref to the multi-task learning model in "[2] Convex multi-task feature learning"

clear

rng(1)

% Generates the group between features; N features grouped into K groups
% with group size gs and overlapping size os


T  = 20;     % 20 tasks
K = 500;     % N features are participate into 500 groups
gs = 6;    % group size: under the assumption that all groups have the same size
os = 2;     % overlapping size
N = (gs-os)*K + os;

Xorig = zeros(N,T);  % X stores all coefficients of all tasks;  column t are the coefficients for task t; row i are coefficients of feature i;

%  particiate the rows of X into K groups with equal group size
grps_rowX = cell(K,1);    
for k = 1:K
    grps_rowX{k} = (gs-os)*(k-1)+1:(gs-os)*(k-1)+gs;
end

Irp = randperm(K);
J = Irp(1:20);       % 20 uniformly active groups of features for every task

lgnm_xorig = 0;    % the approximate latent group norm of x
alpha = 0.2;    % the sparsity ration in each group
for j = 1:length(J)
    Xorig( grps_rowX{J(j)}, : ) = sprandn( gs, T, alpha);
    lgnm_xorig = lgnm_xorig + norm(Xorig( grps_rowX{J(j)}, : ), 'fro');
end

%the true solution of the multi-task learning model min ||Phi*x-b||^2/2 + mu*||x||_1  subject to Kappa(x)<= sigma
xorig = reshape(Xorig, N*T, 1);   % vector of size (N*T, 1);

% % generate Phi_t's and b_t's
m = 500;   % the number of observations for each task; The observation matrix Phi_t is of size (m, N);
nf = 0.01;    % noise factor on the observations; 0.1 and 0.05 is hard: rec_err is about 0.12 even m=800,1000
Phi_all = cell(T,1);
b_all = cell(T,1);
L_all = zeros(T, 1);
for t = 1:T
    % generate the observation matrix for task t
    Phi_t  = randn(m, N);
    for p = 1:N
        Phi_t(:, p) = Phi_t(:, p)/norm(Phi_t(:,p));        % slightly different from that in the reference [1]; they use normal distribution N(0, 1/250*I)
    end
    Phi_all{t,1} = Phi_t;
    L_t = svds(Phi_t, 1)^2;
    L_all(t) = L_t;
    error = nf * rand(m,1);     % Gaussian noise
    b_t = Phi_t*Xorig(:, t) + error;
    b_all{t,1} = b_t;
end


% set the Group partitions for x: aggregating the rows of Xorig that were originally in Grps_rowX
gs_large = gs * T;   % group size of each group in x 
% Grps = zeros(K, gs_large);
Grps = cell(K, 1);
for k = 1: K
    itmp = zeros(gs_large, 1);
    for t = 1:T
        itmp( (t -1)*gs+1: t*gs ) = grps_rowX{k} + (t -1)*N;
    end
    Grps{k} = itmp;
end

f_args = cell(4,1);
f_args{1,1} = Phi_all;
f_args{2,1} = b_all;
f_args{3,1} = T;
f_args{4,1} = N;

n = N*T;

opts.verbose_freq = 1000;
opts.maxiter = 10000;
% sigma_A = svds(A, 1);
% opts.L_A = sigma_A^2;
opts.L_f = max(L_all);
opts.xinit = zeros(n,1);


c1 = 0.95*lgnm_xorig;
mu = 0.10;        % when the nf is small, can choose mu about 0.1 and m =500; larger m does not help
[x, y, iter, history] =  FW_sparselgl(@f_mtl, f_args, n, mu, c1, K, Grps, opts);

rel_err = norm(xorig - x)/max(norm(xorig),1);
X = reshape(x, N, T);
nnz_x = sum(abs(x)>1e-6);
nnz_orig = sum(abs(xorig)>1e-6);

figure;
plot(1:n, xorig, 'bo',1:n, x, 'r*');
xlim([0,n])

figure;
subplot(1,2,1)
spy(X)
subplot(1,2,2)
spy(Xorig)

fval_rec = history.fval_rec;
figure;
plot(1:iter, fval_rec, 'b-', 'LineWidth', 1.5);
xlabel('Iterates');
ylabel('function value');

lc_res_rec = history.lc_res_rec;
figure;
semilogy(1:iter, lc_res_rec, 'b-', 'LineWidth', 1.5);
xlabel('Iterates');
ylabel('log(x-y)')


% gap_rec = history.FWgap_rec;
% figure
% semilogy(1:iter, gap_rec, 'b-', 'LineWidth', 1.5);
% xlabel('Iterates');
% ylabel('log(FW gap)');









