clear
clc
close all


addpath('./HC-CE-Algorithm')


Nt = 64;
Nr = 2;
Nrf = 2;
Ns = 2;   % the number of data streams
Nloop = 1000000;
L=3;
Lest = 3; 


file=load('codebook_bs.mat');
codebook_bs=file.MO;
file=load('codebook_ms.mat');
codebook_ms=file.DBF;


pnr_array=0;
for pnr_id=1:1:length(pnr_array)
    pnr=pnr_array(pnr_id);
    [pcsi, ecsi,data,index_bs,index_ms,nmse] = channel_estimation_train(pnr,Nloop, L, Lest,Nrf, Nt, Nr,codebook_bs,codebook_ms);
    dir=['./dataset/pnr',num2str(pnr),'_train.mat'];
    save(dir,'pnr','pcsi','ecsi','index_bs','index_ms','data','nmse')
end


