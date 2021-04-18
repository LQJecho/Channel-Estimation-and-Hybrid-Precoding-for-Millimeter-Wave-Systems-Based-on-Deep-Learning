function [V_FD, W_FD] = FD_method(H,Ns)
% traditional SVD algorithm for rate maximization
[U,~,V] = svd(H);
V_FD = V(:,1:Ns);
%power constraint
V_FD = V_FD / norm(V_FD,'fro');
W_FD=U(:,1:Ns);
W_FD=W_FD/norm(W_FD,'fro');

