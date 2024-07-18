function [descent, fw_gap, feas] = FW_HP_subp(g, e, poly, y_now, c_D, whole_deg)
% This solves
% min_{y, theta}  <g, x - y_now>
% s.t.  y \in \kappastar, <e, y> <= c_D
% where y_now is the current iterate,
% \kappastar is a convex closed cone related to poly,
% and whole_deg is a degree related to poly.

%calculate eigenvalues of g
eig_g = eigH(g,e,poly);
min_eig_g = min(eig_g);
feas = false;

%solve subproblem and calculate descent direction
if min_eig_g > -zero_eps % If g is included in Lambda(p,e)
    feas = true;
    descent = -y_now;
else
    z = g - min_eig_g*e;
    eig_z = eig_g - min_eig_g; % eig_z is the list of eigenvalues of z
    mult = nnz(abs(eig_z) < zero_eps); %mult is the multiplicity of z
    normal_vec = real(grad_deriv_poly(z,mult-1,poly,whole_deg,e));
    descent = c_D*normal_vec/dot(e,normal_vec) - y_now;
end

fw_gap = -dot(g,descent);