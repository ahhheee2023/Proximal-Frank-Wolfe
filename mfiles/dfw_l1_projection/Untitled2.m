clear;

addpath('.\DDS')
DDS_startup;

% Generate points to project
num_points = 1;
DDS_tol = 1e-8;
n = 10;
deriv_num = 2;

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
        fprintf("Discarded a test point because it is too close to being feasible\n")
    end
    fprintf("Min eigenvalue of the %dth point: %g \n",k,min_eig_dk)
    d(:,k) = dk;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% measure running time of DDS
fprintf('Start DDS experiments \n\n')

% set up
runtime_DDS = zeros(num_points,1);
x_DDS = cell(1,num_points);
y_DDS = cell(1,num_points);
obj_vals_DDS = cell(1,num_points);
itr_DDS = cell(1,num_points);

% measure running time
for k = 1:num_points
    fprintf('%d-th point\n\n',k)
    %%%%% build the problem %%%%%
    u = d(:,k);
    
    % add the constraint ||x-u||_1 -t <= 0
    cons{1,1}='TD';
    cons{1,2}={[4 n], ones(1,n), ones(1,n)};
    A{1,1}=[-1, zeros(1,n);
        zeros(n,1), eye(n)];
    b{1,1}=[0; -u];
    
    %make poly for hyperbolicity cone constraint in DDS
    combi = nchoosek(1:n,n-deriv_num);
    row = size(combi,1);
    poly = zeros(row,n+1);
    for i = 1:row
        for j = combi(i,:)
            poly(i,j) = 1;
        end
    end
    poly(:,n+1) = ones(row,1);
    % DDS paper suggests using sparse matrices (section 11.1 in their MPC paper),
    % so we sparsify the matrix if it has 25% or less nonnzeros
    %
    if nnz(poly)/numel(poly) <= 0.25
        poly = sparse(poly);
    end
    % adding HB constraint
    cons{2,1}='HB';
    cons{2,2}=[n];
    cons{2,3}=poly;
    cons{2,4}='monomial';
    cons{2,5}=[ones(n,1)];
    A{2,1}=[zeros(n,1),eye(n)];
    b{2,1}=[zeros(n,1)];
    
    % set cost vector c
    c=[1; zeros(n,1)];
    
    OPTIONS.tol = DDS_tol;
    
    [x,y1,info]=DDS(c,A,b,cons,OPTIONS);
    x_DDS{k} = x(2:end);
    y_DDS{k} = y1;
    runtime_DDS(k) = info.time;
    itr_DDS{k}(end+1) = info.iter;
    obj_vals_DDS{k} = dual_obj_value(y1, b,cons);
end



%% dual-Frank-Wolfe method
fprintf('\n start dual-Frank-Wolfe method\n\n')
% set up
clearvars poly
e = ones(n,1);

% make poly for hyperbolicity cone constraint
combi = nchoosek(1:n,n-deriv_num);
row = size(combi,1);
poly{1}.type = "matrix";
poly{1}.mat = zeros(row,n+1);
for i = 1:row
    for j = combi(i,:)
        poly{1}.mat(i,j) = 1;
    end
end
poly{1}.mat(:,n+1) = ones(row,1);


%%measure runtime
%prepare data strorages for all datasets d
x_FW = zeros(n,num_points);
y_FW = zeros(n,num_points);
theta_FW = zeros(n,num_points);
itr_FW = zeros(1,num_points);
gap_FW = cell(1,num_points);
feas_FW = cell(1,num_points);
runtime_FW = cell(1,num_points);
history_FW = cell(1,num_points);
obj_vals_FW =zeros(1,num_points);


%measure runtime for each d
for k = 1:num_points
    fprintf('\n %d-th point\n\n',k)
    clearvars opts
    opts.y0 = zeros(n,1);
    %make problem
    b = -d(:,k);
    step.rule = "Lipschitz";
    step.L = 1;
    opts.step = step;
    grad = @(x) (x-b);
    
    %opts.zero_eps = 1e-1;
    %opts.verbose_freq = 10;
    
    %stopping criteria
    opts.stop.maxitr = Inf;
    opts.stop.feas_check = true;
    
    %It is necessary to stop the algorithm when the FW gap is too small or is
    %zero, since this may lead to numerical problems.
    %Still, we want to algorithm to keep going until we reach the desired
    %precision.
    %Setting it to 1e-12 ensures that it will rarely (or never) stop
    %because of the FW gap stopping criterion.
    opts.stop.gap_tol = 1e-12;
    
    %in order to save time, we stop the experiment if an iterate is
    %already within min(ratio_list) of the objective value obtained by DDS.
    %     opts.stop.f = @(x) (0.5*norm(x-d(:,k)).^2);
    %     opts.stop.obj_thresh = min(obj_vals_DDS{k})*(1+min(ratio_list));
    opts.verbose_freq = 100;
    opts.stop.max_time = runtime_DDS(k);
%     opts.stop.max_time = inf;
    
    c = sum(abs(e));    % since ||y||_\infty <= 1, we have <e, y> <= ||e||_1*||y||_\infty = ||e||_1
    if dot(e,opts.y0) > c, c = dot(e,opts.y0); end
    
%     [y_FW(:,k),theta_FW(:,k), x_FW(:,k), status_FW, history_FW{k}] = FW_HP_prox(d(:,k), poly, e, c, opts);
%     
%     %calculate objective values
%     obj_vals_FW(k) = y_FW(:,k)'*d(:,k);
%     
%     y_FW_rec = history_FW{k}.y_rec;
%     y_diff = [];
%     for i = 2:size(y_FW_rec, 2)
%         y_diff = [y_diff; norm(y_FW_rec(:,i) - y_FW_rec(:, i-1))];
%     end
%     poly{1}.type = "symmetric";
%     poly{1}.deg = n-deriv_num;
%     poly{1}.index = 1:n;
%     fprintf('Check feasiblity: the minimal eigvalue of x is %.5e\n',min(eigH(x_FW(:,k),e,poly)));
    
%     %% show the DDS solution and the FW solution
%      [x_DDS{1,1}, x_FW]
     
         [x_FW_rec,y_FW_rec,itr_FW_rec,gap_FW_rec,feas_FW_rec,runtime_FW_rec] = proxFW_HP_exp(u, poly, e, c, opts);

    
end




% %% measure running time of dual-Frank-Wolfe method using eleSym
% fprintf('\n start dual-Frank-Wolfe method using eleSym\n\n')
% % remake poly using "symmetric" option
% clearvars poly
% poly{1}.type = "symmetric";
% poly{1}.deg = n-deriv_num;
% poly{1}.index = 1:n;
% 
% %%measure runtime
% %prepare data strorages for all datasets d
% x_FW_ele = zeros(n,num_points);
% y_FW_ele = zeros(n,num_points);
% theta_FW_ele = zeros(n,num_points);
% itr_FW_ele = zeros(1,num_points);
% gap_FW_ele = cell(1,num_points);
% feas_FW_ele = cell(1,num_points);
% runtime_FW_ele = zeros(1,num_points);
% obj_vals_FW_ele = zeros(1,num_points);
% history_FW_ele = cell(1,num_points);
% 
% 
% %measure runtime for each d
% for k = 1:num_points
%     fprintf('\n %d-th point\n\n',k)
%     % %     clear("opts");
%     clear opts
%     opts.y0 = zeros(n,1);
%     %make problem
%     c = sum(abs(e));    % since ||y||_\infty <= 1, we have <e, y> <= ||e||_1*||y||_\infty = ||e||_1
%     if dot(e,opts.y0) > c, c = dot(e,opts.y0); end
%     
%     %opts.zero_eps = 1e-1;
%     opts.verbose_freq = 100;
%     
%     %stopping criteria
%     opts.stop.maxitr = Inf;
%     
%     %It is necessary to stop the algorithm when the FW gap is too small or is
%     %zero, since this may lead to numerical problems.
%     %Still, we want to algorithm to keep going until we reach the desired
%     %precision.
%     %Setting it to 1e-12 ensures that it will rarely (or never) stop
%     %because of the FW gap stopping criterion.
%     opts.stop.gap_tol = 1e-12;
%     
%     %     opts.stop.feas_check = true;
%     %     %in order to save time, we stop the experiment if an iterate is
%     %     %already within min(ratio_list) of the objective value obtained by DDS.
%     %     opts.stop.f = @(x) (0.5*norm(x-d(:,k)).^2);
%     %     opts.stop.obj_thresh = min(obj_vals_DDS{k})*(1+min(ratio_list));
%     opts.stop.max_time = runtime_DDS(k);
%     
%     [y_FW_ele(:,k),theta_FW_ele(:,k), x_FW_ele(:,k), status_FW_ele, history_FW_ele{k}] = FW_HP_prox(d(:,k), poly, e, c, opts);
%     
%     %calculate objective values
%     obj_vals_FW_ele(k) = y_FW_ele(:,k)'*d(:,k);
%     
%     y_FW_rec_ele = history_FW_ele{k}.y_rec;
%     y_diff_ele = [];
%     for i = 2:size(y_FW_rec_ele, 2)
%         y_diff_ele = [y_diff_ele; norm(y_FW_rec_ele(:,i) - y_FW_rec_ele(:, i-1))];
%     end
%     fprintf('Check feasiblity: the minimal eigvalue of xguss is %.5e\n',min(eigH(x_FW_ele(:,k),e,poly)));
% end


