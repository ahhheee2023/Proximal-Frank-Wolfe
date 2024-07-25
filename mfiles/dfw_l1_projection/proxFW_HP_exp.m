function[x_rec,y_rec,itr,gap_rec,feas_rec,time_rec] = proxFW_HP_exp(u, poly, e, c_D, opts)
%This function solves the following problem by the dual proximal Frank-Wolfe method.
% min  \|x-u\|_1
% s.t. x in Lambda(p,e)
% where  Lambda(p,e) (included in R^m) is a pointed hyperbolicity cone associated with the hyperbolic polynomial 'poly'
%       e (in R^m) is a vector in ri(Lambda(p,e))
% Actually, he proximal Frank-Wolfe method solves
% min_{y, theta}  <u, y>
% s.t. ||y||_\infty <= 1,  theta\in K^*
%       y = theta
%
% This function is built for benchmarking purposes and 
% returns all the iterates produced by the method and,
% therefore, is quite resource intensive. For a more memory efficient
% option version, please use FW_HP.

%Input[datatype]
% u -   m by 1 vector, the point to be projected
% poly - the set of polynomials from R^m to R [structure variables]
%       poly{1}*...*poly{end} represents the whole polynomial.
%       poly{i} is represented by a matrix or an oracle. There is also a special option for dealing with elementary symmetric polynomials.
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
% e - [m by 1 vector], the direction vector
% c_D - real number satisfying <e, x> <= c_D (Assumption 1 for the subproblem in Bruno's paper)
% opts - contains the following structure, default values are in ():
%       y0 - starting point [m by 1 vector] (0)
%       feas_check - [logical] (true) If feas_check is true, the stopping
%                    criteria are judged only when the x_i is feasible.
%       step - specifying step size rule[structure variables](diminishing step size rule).
%              Detailed information is in stepsize.m
%       zero_eps - margin used for judging equalities and inequalities. [real number] (1e-8)
%       verbose_freq - [positive integer] (Inf)
%                      If verbose_freq is specified, some informations are
%                      displayed every opts.verbose_freq iterations.
%       stop - parameters for setting the stopping criteria [structure variables]
%              If one of the criteria described below is achieved, this
%              program stops.
%              gap_tol - FW gap tolerance [real number] (-Inf)
%              maxitr - max number of iterations [positive integer] (10)
%              max_time - max time / unit is second [real number] (Inf)
%              obj_thresh - tolerance of objective value [real number] (-Inf)
%                        opts.stop.f must be given to specify this variable.

%Output[datatype]
%x - points in primal space   x(:,i) corresponds to x_i in algorithm [m by iter matrix]
%y - points in dual space  y(:,i) corresponds to y_i in algorithm [m by iter matrix]
%itr - the number of iterations performed by the solver [positive integer]
%gap - Frank-Wolfe gap gap(i) corresponds to Frank-Wolfe gap at y_i [m by iter matrix]
%feas - feas(i) is 1 if x_i is feasible and 0 otherwise [1 by iter matrix]
%time - time ellapsed after the i-th iteration [1 by iter matrix]

%% start of running time measurement
tic;

%% preprocessing

% options
m = length(u);

if isfield(opts,'y0'), y0 = opts.y0; else y0 = zeros(m,1); end 
if isfield(opts,'feas_check'), feas_check = opts.feas_check; else feas_check = true; end
if isfield(opts,'zero_eps'), zero_eps = opts.zero_eps; else zero_eps = 1e-8; end
if isfield(opts,'verbose_freq')
    verbose_freq = opts.verbose_freq;
    fprintf(" Iteration          Time              FW_gap         min_eig    \n")
    fprintf("----------------------------------------------------------------\n")
else verbose_freq= Inf; end
% stopping criteria
if isfield(opts,'stop') == false, opts.stop = {}; end
if isfield(opts.stop,'gap_tol'), gap_tol = opts.stop.gap_tol; else gap_tol = 1e-2; end
if isfield(opts.stop,'maxitr'), maxitr = opts.stop.maxitr; else maxitr = 5000; end
if isfield(opts.stop,'max_time'), max_time = opts.stop.max_time; else max_time = Inf; end
if isfield(opts.stop,'obj_thresh')
    if ~isfield(opts.stop,'f')
        error('To specify opts.stop.obj_val, set opts.stop.f.');
    end
    obj_thresh = opts.stop.obj_thresh;
else
    obj_thresh = -Inf;
end

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
gap_rec = [];
y_rec = [];
theta= y0;
x_rec = [];
time_rec = [];
feas_rec = [];

%% main algorithm
itr = 1;
while(1)
    beta_t = sqrt(itr);
    theta_mul = beta_t * theta;
    y_mul = min( max(theta_mul-u, -beta_t), beta_t );
    y = y_mul / beta_t;
    y_rec = [y_rec, y];
    
    g = theta_mul - y_mul;
    x = g;
    x_rec = [x_rec, x];
        
    %calculate eigenvalues of g
    eig_g = eigH(g,e,poly);
    min_eig_g = min(eig_g);   % since x = g; this can be used to check feasiblity
    
    % solve the FW subproblem
    % min  <g, theta>  
    % s.t.   theta \in K^*, <e, theta> <= c_D
    % where g = beta_t*(theta_t - y_{t+1}), c_D = ||e||_1
    if min_eig_g > -zero_eps  % If g is included in Lambda(p,e) 
        feas = 1;
        descent = -theta;
    else
        feas = 0;
        z = g - min(eig_g)*e;
        eig_z = eig_g - min_eig_g; % eig_z is the list of eigenvalues of z
        mult = nnz(abs(eig_z) < zero_eps); %mult is the multiplicity of z
        normal_vec = real(grad_deriv_poly(z,mult-1,poly,whole_deg,e));
        descent = c_D*normal_vec/dot(e,normal_vec) - theta;
    end
    feas_rec = [feas_rec; feas];
    
    %calculate the Frank-Wolfe gap and measure time
    gap = - (g'*descent);
    gap_rec = [gap_rec; gap];
    time_crnt = toc;
    time_rec = [time_rec; time_crnt];

    %display intermediary results
    if rem(itr,verbose_freq)==0
        str_time = sprintf("%.5f",time_crnt);
        str_gap = sprintf("%.5e",gap);
        str_itr = sprintf("%d",itr);
        str_min_eig = sprintf("%.5e",min_eig_g);
        fprintf(blanks(10-strlength(str_itr))+str_itr+'   |  ' ...
        +blanks(11-strlength(str_time))+str_time+'   |   '+blanks(13-strlength(str_gap))+str_gap ...
        +'   |   '+blanks(13-strlength(str_min_eig))+str_min_eig+ '\n')
    end
    
    %If one of the stopping criteria is satisfied, stop
    if itr>=maxitr, fprintf("Reached max iteration.\n"), break, end
    if time_rec(end) > max_time, fprintf("Reached max time.\n"), break, end
    if feas  ||  ~feas_check
        f_t = sum(abs(x-u));
        if ( gap < gap_tol || f_t < obj_thresh),  fprintf("The FW gap or the objective value is small enough. \n"),  break, end
    end
    
    %update
    alpha = 1/(itr+2);
    theta = theta + alpha*descent;
    itr = itr + 1;
end
end