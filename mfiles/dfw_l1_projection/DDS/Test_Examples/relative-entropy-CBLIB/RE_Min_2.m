%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% In this example, we are minimizing the relative entropy of two vectors 
%% x and y of length m, such that the entorpy of each of them is bounded. 
%%                           min  sum(rel_entr(x,y)
%%                           s.t.  -sum(entr(x)) <= 4
%%                                 -sum(entr(x)) <= 3
%% For the relative entropy, we use RE type constraints, and for the entorpy we use TD type constraints.
%% Note that we add a variable to take the objective function into constraints. 
%% If feasible = false, the problems has a linear constraint to make it infeassible. 

%% Copyright (c) 2020, by 
%% Mehdi Karimi
%% Levent Tuncel
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

feasible = true;

clearvars c b A cons;
m=10;

% adding vector relative entropy
cons{1,1}='RE';  cons{1,2}=[2*m+1];
A{1,1}=[zeros(1,2*m) -1; zeros(m) eye(m) zeros(m,1) ; eye(m) zeros(m,m+1)];
b{1,1}=[0; zeros(2*m,1)];

% adding 2-dim constraintsc
cons{2,1}='TD';  
cons{2,2} = {[3 m], ones(m,1); [3 m], ones(m,1)};
A{2,1}=[zeros(1,2*m) 0; zeros(m) eye(m) zeros(m,1); ...
        zeros(1,2*m) 0; eye(m) zeros(m) zeros(m,1)];
b{2,1}=[-4;zeros(m,1);-3;zeros(m,1)];


if ~feasible   %  If feasible = false then this LP constraint will be added and the problem becomes infeasible.
% adding LP constraint 
cons{3,1}='LP';
cons{3,2}=[1];
A{3,1}=[-1 zeros(1,2*m)];
b{3,1}=[-1];
end


c=[zeros(2*m,1);1];
[x,y]=DDS(c,A,b,cons);

%%%%%%%%%%%%%% CVX code for comparison
% cvx_begin
% variables x_CVX(m) y_CVX(m)
% minimize sum(rel_entr(x_CVX,y_CVX))
% subject to 
%     -sum(entr(x_CVX)) <= 4
%     -sum(entr(y_CVX)) <= 3
%     if ~feasible
%        x_CVX(1) <= -1
%     end
% cvx_end

