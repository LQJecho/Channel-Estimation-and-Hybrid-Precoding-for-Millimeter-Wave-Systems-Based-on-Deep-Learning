function W_RF  = yuweiA2(V_D,V_RF, Vn, H, Nrf, Nr)

W_RF = ones(Nr,Nrf);
F = H*V_RF*V_D*V_D'*V_RF'*H';
g = 1/Nr;
a = g/Vn;

for Nloop = 1:10
    for j = 1:Nrf
        VRF = W_RF;
        VRF(:,j)=[];
        C = eye(Nrf-1)+a*VRF'*F*VRF;
        G = a*F-a^2*F*VRF*C^(-1)*VRF'*F;
        for i = 1:Nr
            for l = 1:Nr
                if i~=l
                    x(l)=G(i,l)*W_RF(l,j);
                end
            end
            n = sum(x);
            if n ==0
                W_RF(i,j)=1;
            else
                W_RF(i,j)=n/abs(n);
            end
        end
    end
end