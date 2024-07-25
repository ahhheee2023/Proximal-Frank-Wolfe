clearvars

%% parameters
% problem settings
n=20; deriv_num=10; % Figure 1 in the paper

%n=30; deriv_num=15; % Figure 2 in the paper. Remove the comment marker to
% display the plot associated to Figure 2
max_time = 100;
num_points = 1;

rng(1); %fix seed
d = zeros(n,num_points);
poly{1}.type = "symmetric";
poly{1}.deg = n-deriv_num;
poly{1}.index = 1:n;
e = ones(n,1);
for k = 1:num_points
    dk = randn(n,1);
    min_eig_dk = min(eigH(dk,e,poly));
    %Discard test points that are too close to the cone.
    while (min_eig_dk > -1e-4 )
        dk = randn(n,1);
        min_eig_dk = min(eigH(dk,e,poly));
        fprintf("Discarded a test point because it is too close to being feasible\n");
    end
    fprintf("Min eigenvalue of the %dth point: %g \n",k,min_eig_dk);
    d(:,k) = dk;
end

%% measure running time of dual-Frank-Wolfe method using eleSym
fprintf('\n start dual-Frank-Wolfe method using eleSym\n\n')
% set up
clearvars opts
e = ones(n,1);
opts.y0 = zeros(n,1);
opts.verbose_freq = 1000;


for k = 1:num_points
    % measure runtime
    % make problem
    c = sum(abs(e));
    if dot(e,opts.y0) > c, c = dot(e,opts.y0); end
    
    %stopping criteria
    opts.stop.maxitr = Inf;
    opts.stop.gap_tol = -Inf;
    opts.stop.feas_check = true;
    opts.stop.max_time =max_time;
    
    [y, theta, x, status, history] = FW_HP_prox(d(:,k), poly, e, c, opts);
    
    %calculate objective values
    %     obj_vals_FW = sum( abs( x_FW-d(:,k) ) );
    obj_vals_FW = y'*d(:,k);
    
    gap_rec = history.gap_rec;
    figure;
    semilogy(1:length(gap_rec), gap_rec, 'b-', 'LineWidth', 1.5);
end


