%==========================================================================
% huangwei 20170731
%==========================================================================
clear,clc,
NAME = {'s_huangwei','s_lianlian',...
        's_bailin','s_xiangchen',...
        's_zhengzifeng'};

%--------------------------------------------------------------------------
% 提取视觉区信号
for sub = 1:1
    path = fullfile('F:\NISD_HW\',NAME{sub},'\Vnii');
    [vl] = rest_ReadNiftiImage(fullfile(path,'flh_occ.nii'));  % 读取 nii
    [vr] = rest_ReadNiftiImage(fullfile(path,'frh_occ.nii'));  % 读取 nii
    occ  = double(vl|vr);
    [V1]  = double(rest_ReadNiftiImage(fullfile(path,'flh_V1.nii'))|rest_ReadNiftiImage(fullfile(path,'frh_V1.nii')));
    [V2d] = double(rest_ReadNiftiImage(fullfile(path,'flh_V2d.nii'))|rest_ReadNiftiImage(fullfile(path,'frh_V2d.nii')));
    [V2v] = double(rest_ReadNiftiImage(fullfile(path,'flh_V2v.nii'))|rest_ReadNiftiImage(fullfile(path,'frh_V2v.nii')));
    [V2] = double(V2d|V2v);
    [V3v] = double(rest_ReadNiftiImage(fullfile(path,'flh_V3v.nii'))|rest_ReadNiftiImage(fullfile(path,'frh_V3v.nii')));
    [V3d]  = double(rest_ReadNiftiImage(fullfile(path,'flh_V3d.nii'))|rest_ReadNiftiImage(fullfile(path,'frh_V3d.nii')));
    [V3] = double(V3d|V3v);
    [LVC] = double(V1|V2|V3);
    [HVC] = occ - double(LVC & occ);
    [ROI(:,1),ROI(:,2),ROI(:,3)]=ind2sub(size(occ),find(occ>0));
    ROI(:,4) = occ(find(occ>0));
    ROI(:,5) = V1(find(occ>0));
    ROI(:,6) = V2(find(occ>0));
    ROI(:,7) = V3(find(occ>0));
    ROI(:,8) = LVC(find(occ>0));
    ROI(:,9) = HVC(find(occ>0));
    clear v* h* i
    ROI_name = {'x','y','z',...
              'OCC','V1',...
              'V2','V3',...
              'LVC','HVC'};
    % for i=4:length(ROI_name)
    %     path = fullfile('B:\DATA_MRI_3T\',NAME{k},'\Retina_samsrf_pRF');
    %     [v] = rest_ReadNiftiImage(fullfile(path,['f_',ROI_name{i},'.nii']));  % 读取 nii
    for i=4:4
        V = ROI(:,i);
        temp = find(ROI(:,i)>0);
        loc = ROI(temp,1:3);
        name = strcat(NAME{sub},'_location_',ROI_name{i});
        path = fullfile('F:\NISD_HW\',NAME{sub},'\',NAME{sub},'\',ROI_name{i});
        mkdir(path); 
        %cd(path)
        save(name,'loc');
        clear loc name
    end
    clear v h i path 
    %--------------------------------------------------------------------------
    % 提取自然图像重构数据的视觉区信号
    path = fullfile('F:\NISD_HW\',NAME{sub});
    doc=dir(fullfile(path,'\NIR_fmri_raf\','*')); doc_name={doc.name}';
    for m = 4:4
        %region = rest_ReadNiftiImage(fullfile(path,'Retina_samsrf_pRF',['f_',ROI_name{m},'.nii'])); 
        %load(fullfile(path,NAME{sub},ROI_name{m},[NAME{sub},'_location_',ROI_name{m}]));
        load(fullfile(path,NAME{sub},[NAME{sub},'_location_',ROI_name{m}]));
        for i = 3:length(doc_name)
            occ_Rec = zeros(sum(ROI(:,m)>0),408);
            file=dir(fullfile(path,'\NIR_fmri_raf',doc_name{i},'raNIR_*')); file_name={file.name}';    
            for j = 1:length(file_name)
                d = rest_ReadNiftiImage(fullfile(path,'\NIR_fmri_raf',doc_name{i},file_name{j}));  % 读取 nii
                for n = 1:size(loc,1)
                    occ_Rec(n,j) = d(loc(n,1),loc(n,2),loc(n,3));
                end 
            end
            fprintf('%s %s %s 已完成：%d/%d \n',NAME{sub},ROI_name{m},doc_name{i},j,length(file_name));
            clear d 

    %--------------------------------------------------------------------------
    % 去除异常体素点
%             [e(:,1),e(:,2)]=ind2sub(size(occ_Rec),find(occ_Rec==0)); 
%             %occ_Rec(sub2ind(size(occ_Rec),e(:,1),e(:,2)))= [];
%             %occ_Rec(sub2ind(size(occ_Rec),e(:,1),e(:,2)))= [];
%             writeNPY(occ_Rec,['F:\NISD_HW\',NAME{sub},'\',NAME{sub},'\',ROI_name{m},'\',doc_name{i},'.npy']);
%             Y_new = occ_Rec; %对Y的操作生成Y_new
%             V.fname =[doc_name{i},'.npy'];  %保存文件的文件名
%             V = spm_create_vol(V);
%             spm_write_vol(V, Y_new);
%             clear file* num e 
        end
    end
    clear ROI
end




