%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% An example for minimizing a vector relative entropy function.  

%% Copyright (c) 2020, by 
%% Mehdi Karimi
%% Levent Tuncel
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 clearvars c A b cons;

 beta=7; % for beta=2, the problem becomes infeasible. 
 cons{1,1}='RE';
 cons{1,2}=[5]; % This vector is 2*l+1, where l is the length of vectors in the relative entropy function.

 A{1,1} = [0 0 -1; 0.8 0 0; 1.1 -1.5 0; 2.1 1.3 0; 0 3.9 0];
 b{1,1} = [0;1.3;-3.8;1.9;0];
 
 cons{2,1}='LP';
 cons{2,2}=[1];
 
 A{2,1} = [-1 -1 0];
 b{2,1} = [beta];
 
 
 c=[0;0;1];
 
 [x,y,info]=DDS(c,A,b,cons);
