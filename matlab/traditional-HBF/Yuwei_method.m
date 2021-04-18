function [V] = Yuwei_method( H,Ns,Nrf,Vn,Nt )
% the YUWEI algorithm for both narrowband and broadband
% cite the paper Hybrid digital and analog beamforming design for large-scale antenna arrays
V_RF = yuweiA1(Vn, H, Nrf, Nt);
try
Q = (V_RF'*V_RF);
T = Q^(-0.5);
L = H*V_RF*T;
    [~,D,V] = svd(L);
    [~,IX] = sort(diag(D),'descend');
    M = V(:,IX);
    U = M(:,1:Ns);
   V_D = T*U;
catch
V_D = eye(Nrf);
end
    V_D = V_D/norm(V_RF*V_D,'fro');
    V=V_RF*V_D;
end


