function [y, theta, x, status, history] = FW_HP_prox(u, poly, e, c_D, opts)
%This function solves the following problem by the dual-Frank-Wolfe method.
%min  \|x-u\|_1
%s.t. x in Lambda(p,e)
%where  f : (R^n) to (R or {infty}) is the L1 norm
%       Lambda(p,e) (included in R^m) is a pointed hyperbolicity cone associated with hyperbolic polynomial 'poly'
%       e (in R^m) is a vector in ri(Lambda(p,e))
%

%  Modified based on Bruno's code FW_HP.m

%Input[datatype]
%       u: belongs to (R^m), the point to project
%poly - the set of polynomials from R^m to R [structure variables]
%       poly{1}*...*poly{end} represents the whole polynomial.
%       poly{i} is represented by the matrix or oracle, and special option is prepared for elementary symmetric polynomial.
%       Each representation requires following items.
%       <matrix representation>
%       To use matrix representation, set poly{i}.type "matrix".
%       poly{i}.mat - [(the number of terms) times n matrix] Each row represents a term,
%                     and elements in the row correspond to the exponents of variables.
%       <oracle representation>
%       To use oracle representation, set poly{i}.type "oracle".
%       poly{i}.p - [function handle] oracle for evaluating poly{i} at x
%       poly{i}.deg - [integer] degree of poly{i}
%       poly{i}.grad -[function handle] oracle for evaluating the gradient of poly{i} at x
%       <elementary symmetric polynomial>
%       If (x_i1,...x_ik) is in the hyperbolicity cone associated with the k-th symmetric polynomial,
%       set poly{i}.type = "symmetric", poly{i}.index = [i1,...,ik], and poly{i}.deg = k. If you use this option,
%       (e_i1,...,e_ik) must be (1,...,1).
%
%       Optionally you can give the eigenvalues oracle for poly{i} in
%       poly{i}.eig.
%e - [m by 1 vector]
%c_D - real number satisfying Assumpution 1 [real number]
%opts - contains the following structure variables[datatype], default values are in ():
%       y0 - starting point [m by 1 vector] (0)
%       step - specifying step size rule[structure variables](diminishing step size rule).
%              Detailed information is in stepsize.m
%       zero_eps - margin used for judging equalities and inequalities. [real number] (1e-8)
%       verbose_freq - [positive integer] (Inf)
%                      If verbose_freq is specified, some informations are
%                      displayed every opts.verbose_freq iterations.
%       stop - parameters for setting stopping criteria [structure variables]
%              If one of the criteria described below was achieved, this
%              program stops.
%              gap_tol - tolerance of FW gap [real number] (1e-2)
%              maxitr - max number of iteration [positive integer] (Inf)
%              max_time - max time / unit is second [real number] (10)
%              obj_thresh - tolerance of objective value [real number] (-Inf)
%                        opts.stop.f must be given to specify this variable.
%              f - objective function [function handle]

%Output[datatype]
%y - Best dual solution found: among the iterates that are primal feasible within opt.zero_eps,
% return the dual solution with minimal FW gap. If no such iterate exists, y is
% the dual solution corresponding to a primal solution found with minimal feasibility violation.
%status - contains a structure with the following values:
%
%       status - 1: Obtained a feasible solution with FW gap <= stop.gap_tol
%                2: Obtained a feasible solution but FW gap stop criterion
%                was not satisfied.
%                3: No primal feasible solution found
%       time: total running time.
%       itr: total number of iterations.
%       FW_gap: Best FW_gap found. It is Inf if no primal feasible solution
%       was found
%
%

%% start of running time measurement
tic;
%% preprocessing

fval_rec = [];
y_rec = [];
theta_rec = [];
gap_rec = [];

% options
m = size(u, 1);
if isfield(opts,'y0'), y0 = opts.y0; else y0 = zeros(m,1); end
if isfield(opts,'step'), step = opts.step; else step.rule = 'diminishing'; end
if isfield(opts,'zero_eps'), zero_eps = opts.zero_eps; else zero_eps = 1e-8; end
if isfield(opts,'verbose_freq')
    verbose_freq = opts.verbose_freq;
    fprintf(" Iteration          Time        Fval           FW_gap         min_eig_g    \n")
%     fprintf(" Iteration          Time        Fval           FW_gap        \n")
    fprintf("----------------------------------------------------------------\n")
else verbose_freq=1000; end
% stopping criteria
if isfield(opts,'stop') == false, opts.stop = {}; end
if isfield(opts.stop,'gap_tol'), gap_tol = opts.stop.gap_tol; else gap_tol = 1e-2; end
if isfield(opts.stop,'maxitr'), maxitr = opts.stop.maxitr; else maxitr = Inf; end
if isfield(opts.stop,'max_time'), max_time = opts.stop.max_time; else max_time = 10; end

% preprocessing the polynomial
whole_deg = 0;
for i = 1:length(poly)
    if poly{i}.type == "symmetric"
        poly{i}.p = @(x) (eleSym(x(poly{i}.index),poly{i}.deg));
        poly{i}.grad = @(x) (grad_eleSym(x,poly{i}.index,poly{i}.deg));
        
    elseif poly{i}.type == "matrix"
        poly{i}.p = @(x) (sum(prod(cat(1,x.^((poly{i}.mat(:,1:end-1)).'),(poly{i}.mat(:,end)).'))));
        poly{i}.deg = sum(poly{i}.mat(1,1:end-1));
        grad_poly = cell(1,m);
        for term = 1:size(poly{i}.mat,1)
            for j = 1:m
                if poly{i}.mat(term,j) ~= 0
                    tmp = poly{i}.mat(term,:); tmp(m+1) = tmp(j)*tmp(m+1); tmp(j) = tmp(j)-1;
                    grad_poly{j}(end+1,:) = tmp;
                end
            end
        end
        poly{i}.grad = @(x) (poly_vec_val(x,grad_poly));
    end
    whole_deg = whole_deg + poly{i}.deg;  %whole_deg is the degree of polynomial
end

% set up
gap = Inf; %Frank-Wolfe gap
y = y0;
theta = y0;
best_gap = Inf;
best_eig = -Inf;
best_y = zeros(m,1);
best_theta = zeros(m, 1);
best_x = zeros(m,1);
status.status = 3;

%% main algorithm
itr = 1;
while(1) 
    %     beta_t = sqrt(itr)/m;
    beta_t = sqrt(itr);
    
    theta_mul = beta_t * theta;
    y_mul = min( max(theta_mul-u, -beta_t), beta_t );
%     yold = y;
    y = y_mul / beta_t;
      
    g = theta_mul - y_mul;
%     x = theta_mul - beta_t*yold;
    x = g;
    
    fval_primal = norm(x -u, 1);
    fval_rec = [fval_rec; fval_primal];
    
    %calculate eigenvalues of g
    eig_g = eigH(g,e,poly);
    min_eig_g = min(eig_g);
    feas = false;
    
    %solve the FW subproblem and calculate descent direction of theta
    if min_eig_g > -zero_eps % If g is included in Lambda(p,e)
        feas = true;
        descent = -theta;
    else
        z = g - min_eig_g*e;
        eig_z = eig_g - min_eig_g; % eig_z is the list of eigenvalues of z
        mult = nnz(abs(eig_z) < zero_eps); %mult is the multiplicity of z
        normal_vec = real(grad_deriv_poly(z,mult-1,poly,whole_deg,e));
        descent = c_D*normal_vec/dot(e,normal_vec) - theta;
    end
    
    %calculate the Frank-Wolfe gap
    gap = -dot(g,descent);
    gap_rec = [gap_rec; gap];
    
    %If one of the stopping criteria is satisfied, stop
    if feas
        if (gap <= gap_tol)
            status.status = 1;
            best_x = x;
            best_y = y;
            best_theta = theta;
            best_gap = gap;
            break
        elseif (gap < best_gap)
            status.status = 2;
            best_x = x;
            best_y = y;
            best_theta = theta;
            best_gap = gap;
        end
    elseif status.status == 3 && min_eig_g > best_eig
        best_eig = min_eig_g;
        best_x = x;
        best_y = y;
        best_theta = theta;
    end
    
    if itr >= maxitr
        fprintf("Reached max iteration.\n");
        break;
    end
    if toc >= max_time
        fprintf("Reached max time.\n");
        break;
    end
    
    
    %display intermediary results if verbose_freq is set.
    if rem(itr,verbose_freq)==0
        str_time = sprintf("%.5f",toc);
        str_fval = sprintf("%.5e",fval);
        str_gap = sprintf("%.5e",gap);
        str_itr = sprintf("%d",itr);
        str_min_eig = sprintf("%.5e", min_eig_g);
        fprintf(blanks(10-strlength(str_itr))+str_itr+'   |  ' ...
            +blanks(11-strlength(str_time))+str_time+'   |   '+blanks(13-strlength(str_fval))+str_fval+...
            '   |   '+blanks(13-strlength(str_gap))+str_gap ...
            +'   |   '+blanks(13-strlength(str_min_eig))+str_min_eig+ '\n')
    end
    
    %update
    alpha = 2/(itr+2);       % stepsize
    theta = theta + alpha*descent;
    y_rec = [y_rec, y];
    theta_rec = [theta_rec, theta];
    
    itr = itr + 1;
end
x = best_x;
theta = best_theta;
y = best_y;
status.FW_gap = best_gap;
status.itr = itr;
status.time = toc;
history.fval_rec = fval_rec;
history.y_rec = y_rec;
history.theta_rec = theta_rec;
history.gap_rec = gap_rec;
end