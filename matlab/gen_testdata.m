clear
clc
close all

addpath('./HC-CE-Algorithm')
addpath('./traditional-HBF')


Nt = 64;
Nr = 2;
Nrf = 2;
Ns = 2;   % the number of data streams
Nloop = 1000;
Lest = 3; % Number of phase shifters quantization bits
L=3;

file=load('codebook_bs.mat');
ompcb=file.OMP;
mocb=file.MO;
dbfcb=file.DBF;

file=load('codebook_ms.mat');
codebook_ms=file.DBF;

dir='./dataset/channel_test.mat';
file=load(dir);
pcsi=file.pcsi;

pnr_array=0;
snr_array = -10:2:2;
for pnr_id=1:1:length(pnr_array)
    pnr=pnr_array(pnr_id);
    [ecsi_omp,data,index_bs,index_ms,nmse_omp] = channel_estimation_test(pcsi,pnr,Nloop, L, Lest,Nrf, Nt, Nr,ompcb,codebook_ms);
    [ecsi,data,index_bs,index_ms,nmse] = channel_estimation_test(pcsi,pnr,Nloop, L, Lest,Nrf, Nt, Nr,mocb,codebook_ms);
    
    for snr_id = 1 : length(snr_array)
        noise_power = 1 / 10^(snr_array(snr_id)/10);   % Noise Power
        t1 = clock;
        for  n = 1 : Nloop
            H = squeeze(pcsi(n,:,:));
            
            Hest_omp = squeeze(ecsi_omp(n,:,:));
            [V_omp] = Yuwei_method( Hest_omp,Ns,Nrf,noise_power,Nt );
            rate_omp_array(snr_id, n) = get_rate(V_omp,noise_power,Ns,H);
            
            Hest_mo = squeeze(ecsi(n,:,:));
            [V_mo] = Yuwei_method( Hest_mo,Ns,Nrf,noise_power,Nt );
            rate_mo_array(snr_id, n) = get_rate(V_mo,noise_power,Ns,H);
            
            Hest_dbf = squeeze(ecsi_dbf(n,:,:));
            [V_dbf] = Yuwei_method( Hest_dbf,Ns,Nrf,noise_power,Nt );
            rate_dbf_array(snr_id, n) = get_rate(V_dbf,noise_power,Ns,H);
            
            if (n == 20)
                mytoc(t1, Nloop*length(snr_array));
            end
            
        end
        
    end
    
    rate_omp = real(mean(rate_omp_array,2));
    rate_mo = real(mean(rate_mo_array,2));
    rate_dbf = real(mean(rate_dbf_array,2));
    
    
    dir=['./dataset/pnr',num2str(pnr),'_test.mat'];
    save(dir,'pnr','snr_array','rate_omp','rate_mo','ecsi','pcsi','data','index_bs','index_ms','nmse','nmse_omp')
    
end

