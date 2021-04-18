function [pcsi,ecsi,data,index_bs,index_ms,nmse] = channel_estimation_train(pnr_dB,ITER, Num_paths, Lest,Nrf, Nt, Nr,codebook_bs,codebook_ms)

%% ------------------------System Parameters---------------------------------

BSAntennas_Index=0:1:Nt-1; % Indices of the BS Antennas
Num_BS_RFchains=Nrf; % BS RF chains
MSAntennas_Index=0:1:Nr-1; % Indices of the MS Antennas
Num_MS_RFchains=2;  % MS RF chains

%% ---------------- Channel Estimation Algorithm Parameters------------------

G_BS=96; % Required resolution for BS AoD
G_MS=6; % Required resolution for MS AoA

K_BS=2;  % Number of Beamforming vectors per stage
K_MS=2;

S=floor(log(G_BS/Lest)/log(K_BS)); % Number of iterations

% Beamsteering vectors generation
for g=1:1:G_BS
    AbG(:,g)=sqrt(1/Nt)*exp(1j*(2*pi)*BSAntennas_Index*((g-1)/G_BS));
end
% Am generation
for g=1:1:G_MS
    AmG(:,g)=sqrt(1/Nr)*exp(1j*(2*pi)*MSAntennas_Index*((g-1)/G_MS));
end

%% --------------------------------------------------------------------------

pcsi = zeros(ITER,Nr,Nt);
ecsi = zeros(ITER,Nr,Nt);
error_nmse = zeros(ITER,1);
data=zeros([ITER,(S-1)*2*Lest*Lest+36*3]);
index_bs=zeros([ITER,S*Lest*3]);
index_ms=zeros([ITER,Lest]);

%% ---------------------start estimation------------------------------------------------------

t1 = clock;
for iter=1:1:ITER
    
    data_tmp=zeros([S-1,Lest*2,Lest]); % save recived data during DOA estimation
    data1_tmp=zeros([1,Lest*2*6,Lest]); % save recived data during DOA estimation
    
    if mod(iter,2000)==0
        iter
    end
    
    %% ------------------------------ Channel Generation  -------------------------------------------
    % Channel parameters (angles of arrival and departure and path gains)
    AoD=2*pi*rand(1,Num_paths);
    AoA=2*pi*rand(1,Num_paths);
    alpha=(sqrt(1/2)*sqrt(1/Num_paths)*(randn(1,Num_paths)+1j*randn(1,Num_paths)));
    Channel=zeros(Nr,Nt);
    for path=1:1:Num_paths
        Abh(:,path)=sqrt(1/Nt)*exp(1j*BSAntennas_Index*AoD(path));
        Amh(:,path)=sqrt(1/Nr)*exp(1j*MSAntennas_Index*AoA(path));
        Channel=Channel+sqrt(Nt*Nr)*alpha(path)*Amh(:,path)*Abh(:,path)';
    end
    pcsi(iter,:,:) = Channel;
    pnr=10^(0.1*pnr_dB);
    No=1/pnr;
      
    %% -------------------------------------------------------------------------
    
    %Algorithm parameters initialization
    KB_final=[]; % To keep the indecis of the estimated AoDs
    KM_final=[]; % To keep the indecis of the estimated AoAs
    yv_for_path_estimation=zeros(K_BS*Lest,1); % To keep received vectors
    
    for path=1:1:Lest % An iterations for each path
        KB_star=1:1:K_BS*Lest; % Best AoD ranges for the next stage
        KM_star=1:1:K_MS*Lest; % Best AoA ranges for the next stage
        
        %% --------------------------------------------level=1-----------------------------------
        level=1;
        % Noise calculations
        W_HP=codebook_ms;
        W_HP=reshape(W_HP,[Nr,2*Lest]);
        Noise=W_HP'*(sqrt(No/2)*(randn(Nr,K_BS*Lest)+1j*randn(Nr,K_BS*Lest)));
        F_HP=codebook_bs(level,:,KB_star);
        F_HP=reshape(F_HP,[Nt,2*Lest]);
        Y=W_HP'*Channel*F_HP+Noise;
        
        
        yv=reshape(Y,K_BS*K_MS*Lest^2,1); % vectorized received signal
        data1_tmp(1,:,path)=yv;
        % Subtracting the contribution of previously estimated paths
        for i=1:1:length(KB_final)
            A1=transpose(F_HP)*conj(AbG(:,KB_final(i)+1));
            A2=W_HP'*AmG(:,KM_final(i)+1);
            Prev_path_cont=kron(A1,A2);
            Alp=Prev_path_cont'*yv;
            yv=yv-Alp*Prev_path_cont/(Prev_path_cont'*Prev_path_cont);
        end
        
        % Maximum power angles estimation
        Y_tmp=reshape(yv,K_MS*Lest*K_BS*Lest,1);
        Y=reshape(yv,K_MS*Lest,K_BS*Lest);
        
        %MS
        [val mX]=sort(abs(Y_tmp));
        Max=max(val);
        [KM_max KB_temp]=find(abs(Y)==Max);
        KM_hist(path)=KM_star(KM_max);
        
        %BS
        [~,mX]=sort(abs(Y(KM_max,:)),'descend');
        index_bs_tmp(level,:,path)=KB_star(mX(1:Lest));
        mx_id=1;
        for kk=path:1:Lest
            KB_hist(kk,level)=KB_star(mX(mx_id));
            mx_id=mx_id+1;
        end
        
        % Adjusting the directions of the next stage (The adaptive search)
        for ln=1:1:Lest
            KB_star((ln-1)*K_BS+1:ln*K_BS)=(KB_hist(ln,level)-1)*K_BS+1:1:(KB_hist(ln,level))*K_BS;
        end
        KM_star=KM_hist(path);
        
        %% ---------------------------------------the other levels------------------------------------------------
        
        for level=2:1:S
            
            % Noise calculations
            W_HP=codebook_ms(1,:,KM_star);
            W_HP=reshape(W_HP,[Nr,1]);
            Noise=W_HP'*(sqrt(No/2)*(randn(Nr,K_BS*Lest)+1j*randn(Nr,K_BS*Lest)));
            
            % Received signal
            F_HP=codebook_bs(level,:,KB_star);
            F_HP=reshape(F_HP,[Nt,2*Lest]);
            Y=W_HP'*Channel*F_HP+Noise;
            
            yv=reshape(Y,K_BS*Lest,1); % vectorized received signal
            data_tmp(level-1,:,path)=yv;
            if(level==S)
                yv_for_path_estimation=yv_for_path_estimation+yv;
            end
            
            % Subtracting the contribution of previously estimated paths
            for i=1:1:length(KB_final)
                A1=transpose(F_HP)*conj(AbG(:,KB_final(i)+1));
                A2=W_HP'*AmG(:,KM_final(i)+1);
                Prev_path_cont=kron(A1,A2);
                Alp=Prev_path_cont'*yv;
                yv=yv-Alp*Prev_path_cont/(Prev_path_cont'*Prev_path_cont);
            end
            
            % Maximum power angles estimation
            [~,mX]=sort(abs(Y),'descend');
            index_bs_tmp(level,:,path)=KB_star(mX(1:Lest));
            mx_id=1;
            for kk=path:1:Lest
                KB_hist(kk,level)=KB_star(mX(mx_id));
                mx_id=mx_id+1;
            end
            
                        
            %  Final AoAs/AoDs
            if(level==S)
                KB_final=[KB_final KB_star(mX(1))-1];
                KM_final=[KM_final KM_star-1];
                W_paths(path,:,:)=W_HP;
                F_paths(path,:,:)=F_HP;
            end
            
            % Adjusting the directions of the next stage (The adaptive search)
            for ln=1:1:Lest
                KB_star((ln-1)*K_BS+1:ln*K_BS)=(KB_hist(ln,level)-1)*K_BS+1:1:(KB_hist(ln,level))*K_BS;
            end

            
        end % -- End of estimating the lth path
        
        
    end %--- End of estimation of the channel
    
    %% --------------- Reconstructe the estimated channel------------------
    
    % -----------------------------Estimated angles----------------------
    AoD_est=2*pi*KB_final/G_BS;
    AoA_est=2*pi*KM_final/G_MS;
    
    % ---------------------------Estimated paths--------------------------
    Wx=zeros(Nr,1);
    Fx=zeros(Nt,K_BS*Lest);
    Epsix=zeros(K_BS*Lest,Lest);
    
    for path=1:1:Lest
        Epsi=[];
        Wx(:,:)=W_paths(path,:,:);
        Fx(:,:)=F_paths(path,:,:);
        for i=1:1:length(KB_final)
            A1=transpose(Fx)*conj(AbG(:,KB_final(i)+1));
            A2=Wx'*AmG(:,KM_final(i)+1);
            E=kron(A1,A2);
            Epsi=[Epsi E];
        end
        Epsix=Epsix+Epsi;
    end
    alpha_est=Epsix\yv_for_path_estimation;
    
    %--------------- Reconstructe the estimated channel------------------
    Channel_est=zeros(Nr,Nt);
    for path=1:1:Lest
        Abh_est(:,path)=sqrt(1/Nt)*exp(1j*BSAntennas_Index*AoD_est(path));
        Amh_est(:,path)=sqrt(1/Nr)*exp(1j*MSAntennas_Index*AoA_est(path));
        Channel_est=Channel_est+alpha_est(path)*Amh_est(:,path)*Abh_est(:,path)';
    end
    
    %% ---------------Save data--------------------
    ecsi(iter,:,:) = Channel_est;
    error_nmse(iter)=(norm(Channel_est-Channel,'fro')/norm(Channel,'fro'))^2;
    data_tmp=MyReshape(data_tmp);
    data1_tmp=MyReshape(data1_tmp);
    data(iter,:)=[data_tmp,data1_tmp];
    index_bs(iter,:)=MyReshape(index_bs_tmp);
    index_ms(iter,:)=reshape(KM_hist,[3,1]);
    
end
nmse=mean(error_nmse);
end