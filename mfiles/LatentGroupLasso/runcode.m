clear;

rng(1)

nf = 0.001;  % noise factor
sr = 0.05;  %  sparsity ratio;
index = 1;
gs = 50;   % set the size of all groups equal. group size = 50;
os = 5;     % the size of the overlapping in group i and i+1
K = 100;   % K groups
n = ( (gs-os)*K  + os);
m = floor((gs-os)*K/2)*index;


% % Every groups has 50 elements; the last os indices in {G_i} overlaps the first os indices in {G_{i+1}}
Grps = cell(2,1);
for k = 1:K
    Grps{k} = ( ((gs-os)*(k-1))+1:((gs-os)*(k-1))+ gs)';
end
Irp = randperm(K);

xorig = zeros(n, 1);
lglnm_xorig = 0;
J = Irp(1:floor(K*sr));
sk_orig = zeros(50, floor(K*sr));
for j = 1:length(J)
    sk_orig(:, j) = randn( length(Grps{J(j)} ), 1);
    xorig( Grps{J(j)} ) =  xorig( Grps{J(j)} ) + sk_orig(:, j);    % generate the original signal
    lglnm_xorig = lglnm_xorig + norm(sk_orig);
end

c1 = 0.95*lglnm_xorig;  % the right hand side constant in the constriant kappa(x)<= c1

A = randn(m,n);
for j = 1:n
    A(:, j) = A(:, j)/norm(A(:,j));    % normalization or not?
end
error = nf * (rand(m,1).*rand(m,1) - rand(m,1).*rand(m,1));
b = A*xorig + error;

opts.verbose_freq = 1000;
opts.maxiter = 50000;
L = svds(A, 1);
opts.c2 = L*c1 + norm(b);
opts.xinit = zeros(n,1);

[x, y, iter, history] =  FW_lgl(A, b, c1, K, Grps, opts);

% for k=1:K
%     lglnm_x = 
% end

rel_err = norm(xorig - x)/max(norm(x),1);

fval_rec = history.fval_rec;
gap_rec = history.FWgap_rec;

figure;
plot(1:n, xorig, 'bo',1:n, x, 'r*');
xlim([0,n])

figure;
plot(1:iter, fval_rec, 'b-', 'LineWidth', 1.5);
xlabel('Iterates');
ylabel('function value');

figure
plot(1:iter, gap_rec, 'b-', 'LineWidth', 1.5);
xlabel('Iterates');
ylabel('FW gap');







    