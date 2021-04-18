clc
close all
clear

ITER=10000;
Num_MS_Antennas=2;
Num_BS_Antennas=64;
Num_paths=4;
BSAntennas_Index=0:1:Num_BS_Antennas-1; % Indices of the BS Antennas
MSAntennas_Index=0:1:Num_MS_Antennas-1; % Indices of the MS Antennas

pcsi=zeros(ITER,Num_MS_Antennas,Num_BS_Antennas);
for iter=1:1:ITER
    AoD=2*pi*rand(1,Num_paths);
    AoA=2*pi*rand(1,Num_paths);
    alpha=(sqrt(1/2)*sqrt(1/Num_paths)*(randn(1,Num_paths)+1j*randn(1,Num_paths)));
    
    % Channel construction
    Channel=zeros(Num_MS_Antennas,Num_BS_Antennas);
    for l=1:1:Num_paths
        Abh(:,l)=sqrt(1/Num_BS_Antennas)*exp(1j*BSAntennas_Index*AoD(l));
        Amh(:,l)=sqrt(1/Num_MS_Antennas)*exp(1j*MSAntennas_Index*AoA(l));
        Channel=Channel+sqrt(Num_BS_Antennas*Num_MS_Antennas)*alpha(l)*Amh(:,l)*Abh(:,l)';
    end
    pcsi(iter,:,:) = Channel;
end

save('dataset_L4/channel.mat','pcsi');