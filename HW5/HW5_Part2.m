clear;
clc;
load('TrainingSamplesDCT_8_new.mat');
FG = TrainsampleDCT_FG;
BG = TrainsampleDCT_BG;

sample_BG = size(BG,1);
sample_FG = size(FG,1);

feature = size(BG,2);

img = im2double(imread('cheetah.bmp'));
mask = im2double(imread('cheetah_mask.bmp'));

%read Zig-Zag Pattern.txt file
zz = fopen('Zig-Zag Pattern.txt','r');
zzPat = fscanf(zz,'%d',[8,8])+1;
fclose(zz);

% obtain the DCT of the image
[row,colm] = size(img);
img_zzs = zeros(row-8,colm-8,64);
for i = 1:row-8
    for j = 1:colm-8
        dctImg = dct2(img(i:i+7,j:j+7));
        for x = 1:8
            for y = 1:8
                img_zzs(i,j,zzPat(x,y)) = dctImg(x,y); 
            end
        end
    end
end
[r,m] = size(img_zzs,1,2);
%% Part b)
C = [1,2,4,8,16,32];

% Initialization
pi_BG_1 = zeros(1,1);
mu_BG_1 = zeros(1,feature,1);
sigma_BG_1 = zeros(feature,feature,1,1);

pi_FG_1 = zeros(1,1);
mu_FG_1 = zeros(1,feature,1);
sigma_FG_1 = zeros(feature,feature,1,1);

pi_BG_2 = zeros(2,1);
mu_BG_2 = zeros(2,feature,1);
sigma_BG_2 = zeros(feature,feature,2,1);

pi_FG_2 = zeros(1,1);
mu_FG_2 = zeros(2,feature,1);
sigma_FG_2 = zeros(feature,feature,2,1);

pi_BG_4 = zeros(4,1);
mu_BG_4 = zeros(4,feature,1);
sigma_BG_4 = zeros(feature,feature,4,1);

pi_FG_4 = zeros(4,1);
mu_FG_4 = zeros(4,feature,1);
sigma_FG_4 = zeros(feature,feature,4,1);

pi_BG_8 = zeros(8,1);
mu_BG_8 = zeros(8,feature,1);
sigma_BG_8 = zeros(feature,feature,8,1);

pi_FG_8 = zeros(8,1);
mu_FG_8 = zeros(8,feature,1);
sigma_FG_8 = zeros(feature,feature,8,1);

pi_BG_16 = zeros(16,1);
mu_BG_16 = zeros(16,feature,1);
sigma_BG_16 = zeros(feature,feature,16,1);

pi_FG_16 = zeros(16,1);
mu_FG_16 = zeros(16,feature,1);
sigma_FG_16 = zeros(feature,feature,16,1);

pi_BG_32 = zeros(32,1);
mu_BG_32 = zeros(32,feature,1);
sigma_BG_32 = zeros(feature,feature,32,1);

pi_FG_32 = zeros(32,1);
mu_FG_32 = zeros(32,feature,1);
sigma_FG_32 = zeros(feature,feature,32,1);

for c = C
    pi_BG = rand(1,c);
    pi_BG = pi_BG./sum(pi_BG);
    mu_BG = BG(randperm(sample_BG,c),:);

    pi_FG = rand(1,c);
    pi_FG = pi_FG./sum(pi_FG);
    mu_FG = FG(randperm(sample_FG,c),:);

    sigma_BG = zeros(feature,feature,c);
    sigma_FG = zeros(feature,feature,c);
    for i = 1:c
        sigma_BG(:,:,i) = diag(rand(1,feature)+1e-6);
        sigma_FG(:,:,i) = diag(rand(1,feature)+1e-6);
    end

    % BG EM
    [pi_BG,mu_BG,sigma_BG] = EM(c,BG,pi_BG,mu_BG,sigma_BG);
    % FG EM 
    [pi_FG,mu_FG,sigma_FG] = EM(c,BG,pi_FG,mu_FG,sigma_FG);
    
    if c == 1
       pi_BG_1 = pi_BG;
       mu_BG_1 = mu_BG;
       sigma_BG_1 = sigma_BG;

       pi_FG_1 = pi_FG;
       mu_FG_1 = mu_FG;
       sigma_FG_1 = sigma_FG;
    elseif c == 2
       pi_BG_2 = pi_BG;
       mu_BG_2 = mu_BG;
       sigma_BG_2 = sigma_BG;

       pi_FG_2 = pi_FG;
       mu_FG_2 = mu_FG;
       sigma_FG_2 = sigma_FG;
    elseif c == 4
       pi_BG_4 = pi_BG;
       mu_BG_4 = mu_BG;
       sigma_BG_4 = sigma_BG;
       
       pi_FG_4 = pi_FG;
       mu_FG_4 = mu_FG;
       sigma_FG_4 = sigma_FG;
    elseif c == 8
       pi_BG_8 = pi_BG;
       mu_BG_8 = mu_BG;
       sigma_BG_8 = sigma_BG;
       
       pi_FG_8 = pi_FG;
       mu_FG_8 = mu_FG;
       sigma_FG_8 = sigma_FG;
    elseif c == 16
       pi_BG_16 = pi_BG;
       mu_BG_16 = mu_BG;
       sigma_BG_16 = sigma_BG;
       
       pi_FG_16 = pi_FG;
       mu_FG_16 = mu_FG;
       sigma_FG_16 = sigma_FG;
    else
       pi_BG_32 = pi_BG;
       mu_BG_32 = mu_BG;
       sigma_BG_32 = sigma_BG;
       
       pi_FG_32 = pi_FG;
       mu_FG_32 = mu_FG;
       sigma_FG_32 = sigma_FG;
    end
end
%%
% **************Calculate the Probability of FG and BG for BDR*************
dimensions = [1,2,4,8,16,24,32,40,48,56,64];
PY_BG = sample_BG/(sample_FG+sample_BG);
PY_FG = sample_FG/(sample_FG+sample_BG);

% size =  247*262*11*6
PX_BG = zeros(r,m,length(dimensions),length(C)); 
PX_FG = zeros(r,m,length(dimensions),length(C));

PX_BG(:,:,:,1) = calPX(1,img_zzs,pi_BG_1,mu_BG_1,sigma_BG_1,PY_BG);
PX_FG(:,:,:,1) = calPX(1,img_zzs,pi_FG_1,mu_FG_1,sigma_FG_1,PY_FG);

PX_BG(:,:,:,2) = calPX(2,img_zzs,pi_BG_2,mu_BG_2,sigma_BG_2,PY_BG);
PX_FG(:,:,:,2) = calPX(2,img_zzs,pi_FG_2,mu_FG_2,sigma_FG_2,PY_FG);

PX_BG(:,:,:,3) = calPX(4,img_zzs,pi_BG_4,mu_BG_4,sigma_BG_4,PY_BG);
PX_FG(:,:,:,3) = calPX(4,img_zzs,pi_FG_4,mu_FG_4,sigma_FG_4,PY_FG);

PX_BG(:,:,:,4) = calPX(8,img_zzs,pi_BG_8,mu_BG_8,sigma_BG_8,PY_BG);
PX_FG(:,:,:,4) = calPX(8,img_zzs,pi_FG_8,mu_FG_8,sigma_FG_8,PY_FG);

PX_BG(:,:,:,5) = calPX(16,img_zzs,pi_BG_16,mu_BG_16,sigma_BG_16,PY_BG);
PX_FG(:,:,:,5) = calPX(16,img_zzs,pi_FG_16,mu_FG_16,sigma_FG_16,PY_FG);
%%
PX_BG(:,:,:,6) = calPX(32,img_zzs,pi_BG_32,mu_BG_32,sigma_BG_32,PY_BG);
PX_FG(:,:,:,6) = calPX(32,img_zzs,pi_FG_32,mu_FG_32,sigma_FG_32,PY_FG);
%
% ************************Calculate the PoE*******************************
error_EM = zeros(6,length(dimensions));
error_EM(1,:) = countError(mask,PX_FG(:,:,:,1),PX_BG(:,:,:,1));
error_EM(2,:) = countError(mask,PX_FG(:,:,:,2),PX_BG(:,:,:,2));
error_EM(3,:) = countError(mask,PX_FG(:,:,:,3),PX_BG(:,:,:,3));
error_EM(4,:) = countError(mask,PX_FG(:,:,:,4),PX_BG(:,:,:,4));
error_EM(5,:) = countError(mask,PX_FG(:,:,:,5),PX_BG(:,:,:,5));
error_EM(6,:) = countError(mask,PX_FG(:,:,:,6),PX_BG(:,:,:,6));

figure
hold on;
plot(dimensions,error_EM(1,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(2,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(3,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(4,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(5,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(6,:),'-o','MarkerSize',3,'LineWidth',1.5);
legend('C=1','C=2','C=4','C=8','C=16','C=32');
title('The PoE vs. Dimension With Different Numbers of Components C');
xlabel('Dimensions');
ylabel('PoE')
grid on;
hold off;

%%
function [Pi_n,mu_n,sigma_n] = EM(C,X,Pi,mu,sigma)
    iter = 100; % maximum iteration
    likehood = zeros(1,iter); % log likehood for stopping EM
    jointpdf = zeros(size(X,1),C); 
    mu_n = zeros(C,size(X,2));
    sigma_n = zeros(size(X,2),size(X,2),C);
    % EM
    for h = 1:iter
        % check the log likehood to stop the function once meet the
        % condition
        if h > 1
            if(abs((likehood(h)-likehood(h-1))/likehood(h))<0.01)
                break;
            end
        end

        % E-step
        for i = 1:C
            for j = 1:size(X,1)
                jointpdf(j,i) = sqrt((2*pi)^64*det(sigma(:,:,i)))^(-1)*exp(-(X(j,:)-mu(i,:))/sigma(:,:,i)*(X(j,:)-mu(i,:))'/2)*Pi(i);
            end
        end
        hij = jointpdf ./ sum(jointpdf,2);

        % M-step
        Pi_n = sum(hij,1)/size(X,1);
        for i = 1:C
            mu_temp = 0;
            sigma_temp = 0;
            for j = 1:size(X,1)
                mu_temp = mu_temp + hij(j,i)*X(j,:);
                sig_temp = diag((X(j,:)-mu(i,:))'*(X(j,:)-mu(i,:)));
                sig_temp(sig_temp<1e-5) = 1e-5;
                sigma_temp = sigma_temp + hij(j,i)*diag(sig_temp);
            end
            mu_n(i,:) = mu_temp/sum(hij(:,i),1);
            sigma_n(:,:,i) = sigma_temp/sum(hij(:,i),1);
        end
    end
end

function px = calPX(C,X,Pi,mu,sigma,PY)
    px = zeros(247,262,11);
    dimensions = [1,2,4,8,16,24,32,40,48,56,64];
    for i = 1:length(dimensions)
        dim = dimensions(i);
        for x = 1:247
            for y = 1:262
                Xtemp(1:dim) = X(x,y,1:dim);
                prob = 0;
                for j = 1: C
                    prob = prob + mvnpdf(Xtemp,mu(j,1:dim),sigma(1:dim,1:dim,j))*Pi(j);
                end
                px(x,y,i) = prob;
            end
        end
    end
    px = px.*PY;
end

function errs = countError(mask,p_fg,p_bg)
    pred = zeros(255,270);
    errs = zeros(1,11);
    for j = 1:11
        count = 0;
        for x = 1:247
            for y = 1:262
                if p_bg(x,y,j)<=p_fg(x,y,j)
                    pred(x,y) = 1;
                end

                if pred(x,y) ~= mask(x,y)
                    count = count + 1; 
                end
            end
        end
        errs(j) = count/(255*270);
    end
%     
    figure
    imagesc(pred);
    colormap(gray(255));
end
