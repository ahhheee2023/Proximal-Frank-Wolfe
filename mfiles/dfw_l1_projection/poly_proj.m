function[y,  status, history] = poly_proj(x0,p,e,opts)
%Computes a low-to-medium accuracy L1 projection of x0 onto the hyperbolicity 
%cone given by p and e using the dual Frank-Wolfe method in FW_HP.m.
%By default, x0 is scaled to have 2-norm less or equal than 1 before being passed 
%to the solver and, then, the output is unscaled back. This can 
%be disabled by setting opts.scale = False.
% 
%See poly_proj_examples.m for examples.
%
%Input:
%x0 - [m by 1 vector]
%p - set of polynomials following the format described in FW_HP.m
%e - [m by 1 vector]
%opts - options to be passed to the solver, see opts argument in FW_HP.m.
%Addionally, it is possible to set opts.scale to false in order to disable 
%input scaling.
%
%
%Output:
%x - [m by 1 vector] 
%status: it is the same as the output argument "status" of FW_HP.m:
%       status - 1: Obtained a feasible solution with FW gap <= stop.gap_tol
%                2: Obtained a feasible solution but FW gap stop criterion
%                was not satisfied. In this case, x is the feasible solution 
%                found with the smallest FW gap.
%                3: No primal feasible solution found. In this case, 
%                x is the solution found with the smallest feasibility
%                violation.
%       time: total running time.
%       itr: total number of iterations.
%       FW_gap: Best FW_gap found. It is Inf if no dual feasible solution
%       was found
%

n = size(x0,1);
if nargin < 4
    opts = {};
end
if isfield(opts,'step') == false, opts.step = {}; end
if isfield(opts.step,'rule') == false, opts.step.rule = "Lipschitz"; end
if isfield(opts.step,'L') == false, opts.step.L = 1; end
if isfield(opts,'stop') == false, opts.stop = {}; end
if isfield(opts.stop,'max_time') == false, opts.stop.max_time = 10; end
if isfield(opts,'y0') == false, opts.y0 = zeros(n,1); end   
if isfield(opts,'scale') == false, opts.scale = true; end   

scale_factor = 1;
if opts.scale == true && norm(x0) > 1  
    scale_factor = norm(x0);
end
x1 = x0/scale_factor;    %???  how to scale if we consider L1 projection

c = sum(abs(e));    % since ||y||_\infty <= 1, we have <e, y> <= ||e||_1*||y||_\infty = ||e||_1
if dot(e,opts.y0) > c, c = dot(e,opts.y0); end

% [x,~,status, history] =  FW_HP_prox(x1, p, e, c, opts);
[y, ~, status, history] = FW_HP_prox(x1, p, e, c, opts);
% % x = x*scale_factor;
% x = u 
end

