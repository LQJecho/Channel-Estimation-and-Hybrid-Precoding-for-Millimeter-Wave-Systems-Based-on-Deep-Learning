%功率分配上层功率更高，这个功率分配只是在实际应用中降低中断概率，
%并不影响仿真中的估计结果，因为每一层的相对功率还是一样的，范围选择只针对相对功率大小
function [Pr]=power_allocation(Num_BS_Antennas,Num_BS_RFchains,BSAntennas_Index,G_BS,G_MS,K_BS,K_MS,Num_paths_est,Num_Qbits)

Num_steps=floor(max(log(G_BS/Num_paths_est)/log(K_BS),log(G_MS/Num_paths_est)/log(K_MS)));

% RF codebook
for g=1:1:G_BS
    AbG(:,g)=sqrt(1/Num_BS_Antennas)*exp(1j*(2*pi)*BSAntennas_Index*((g-1)/G_BS));
end

% Generating the hybrid beamforming vectors to obtain the beamformin gain
% of each adaptive hierarchical stage
KB_star=1:1:K_BS*Num_paths_est; % Best AoD ranges for the next stage
KM_star=1:1:K_MS*Num_paths_est; % Best AoA ranges for the next stage

for t=1:1:Num_steps
%Generating the "G" matrix in the paper used to construct the ideal
%training beamforming and combining matrices - These matrices capture the
%desired projection of the ideal beamforming/combining vectors on the
%quantized steering directions

%BS G matrix
G_matrix_BS=zeros(K_BS*Num_paths_est,G_BS);
Block_size_BS=G_BS/(Num_paths_est*K_BS^t);
Block_BS=[ones(1,Block_size_BS)];
for k=1:1:K_BS*Num_paths_est
    G_matrix_BS(k,(KB_star(k)-1)*Block_size_BS+1:(KB_star(k))*Block_size_BS)=Block_BS;
end

% Ideal vectors generation
F_UC=(AbG*AbG')^(-1)*(AbG)*G_matrix_BS';
F_UC=F_UC*diag(1./sqrt(diag(F_UC'*F_UC)));

% Hybrid Precoding Approximation
for m=1:1:K_BS*Num_paths_est
[F_HP(:,m)]=HybridPrecoding(F_UC(:,m),Num_BS_Antennas,Num_BS_RFchains,Num_Qbits);
end

Proj=F_HP(:,1)'*AbG;
G(t)=sum(Proj(1:Block_size_BS))/Block_size_BS;
end

% Optimal power allocation -- according to the paper theorems
Gc=sum(1./G);
Pr=(1/Gc)*1./G;
end
