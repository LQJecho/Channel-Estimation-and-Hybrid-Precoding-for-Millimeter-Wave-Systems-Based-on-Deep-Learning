function new_array=MyReshape(array)
new_array=[];
[S,~,Num_paths]=size(array);
for ii=1:1:Num_paths
    tmp=[];
    for kk=1:1:S
        tmp=[tmp,array(kk,:,ii)];
    end
    new_array=[new_array,tmp];
end