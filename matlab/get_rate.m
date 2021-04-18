function rate = get_rate(V_equal, Vn, Ns, H)
%get the rate (SE) for equivalent V and W
rate = log2(det(eye(Ns) + 1/Vn * H * V_equal * V_equal' * H'));
