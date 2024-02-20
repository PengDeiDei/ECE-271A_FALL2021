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

%% Part a)
C = 8;
dimensions = [1,2,4,8,16,24,32,40,48,56,64];
PY_BG = sample_BG/(sample_FG+sample_BG);
PY_FG = sample_FG/(sample_FG+sample_BG);

% ****************************BG EM****************************************
pi_BG = zeros(C,5);
mu_BG = zeros(C,feature,5);
sigma_BG = zeros(feature,feature,C,5);

for h = 1:5
    pi = randi(1,C);
    pi = pi./sum(pi);
    mu = BG(randperm(sample_BG,C),:);
    sigma = zeros(feature,feature,C);
    
    for i = 1:C
        sigma(:,:,i) = (rand(1,feature)).*eye(feature);
    end
    [pi,mu,sigma] = EM(C,BG,pi,mu,sigma);

    pi_BG(:,h) = pi;
    mu_BG(:,:,h) = mu;
    sigma_BG(:,:,:,h) = sigma;
end

% ****************************FG EM****************************************
% Initialization
pi_FG = zeros(C,5);
mu_FG = zeros(C,feature,5);
sigma_FG = zeros(feature,feature,C,5);

for h = 1:5
    pi = randi(1,C);
    pi = pi./sum(pi);
    mu = FG(randperm(sample_FG,C),:);
    sigma = zeros(feature,feature,C);
    
    for i = 1:C
        sigma(:,:,i) = (rand(1,feature)).*eye(feature);
    end
    [pi,mu,sigma] = EM(C,FG,pi,mu,sigma);

    pi_FG(:,h) = pi;
    mu_FG(:,:,h) = mu;
    sigma_FG(:,:,:,h) = sigma;
end
%%
% **************Calculate the Probability of FG and BG for BDR*************
PX_BG = zeros(r,m,length(dimensions),5);
PX_FG = zeros(r,m,length(dimensions),5);

PX_BG(:,:,:,1) = calPX(C,img_zzs,pi_BG(:,1),mu_BG(:,:,1),sigma_BG(:,:,:,1),PY_BG);
PX_FG(:,:,:,1) = calPX(C,img_zzs,pi_FG(:,1),mu_FG(:,:,1),sigma_FG(:,:,:,1),PY_FG);

PX_BG(:,:,:,2) = calPX(C,img_zzs,pi_BG(:,2),mu_BG(:,:,2),sigma_BG(:,:,:,2),PY_BG);
PX_FG(:,:,:,2) = calPX(C,img_zzs,pi_FG(:,2),mu_FG(:,:,2),sigma_FG(:,:,:,2),PY_FG);

PX_BG(:,:,:,3) = calPX(C,img_zzs,pi_BG(:,3),mu_BG(:,:,3),sigma_BG(:,:,:,3),PY_BG);
PX_FG(:,:,:,3) = calPX(C,img_zzs,pi_FG(:,3),mu_FG(:,:,3),sigma_FG(:,:,:,3),PY_FG);

PX_BG(:,:,:,4) = calPX(C,img_zzs,pi_BG(:,4),mu_BG(:,:,4),sigma_BG(:,:,:,4),PY_BG);
PX_FG(:,:,:,4) = calPX(C,img_zzs,pi_FG(:,4),mu_FG(:,:,4),sigma_FG(:,:,:,4),PY_FG);

PX_BG(:,:,:,5) = calPX(C,img_zzs,pi_BG(:,5),mu_BG(:,:,5),sigma_BG(:,:,:,5),PY_BG);
PX_FG(:,:,:,5) = calPX(C,img_zzs,pi_FG(:,5),mu_FG(:,:,5),sigma_FG(:,:,:,5),PY_FG);
%%
%************************Calculate the PoE*******************************
error_EM = zeros(25,length(dimensions));
% Set 1
error_EM(1,:) = countError(mask,PX_FG(:,:,:,1),PX_BG(:,:,:,1));
error_EM(2,:) = countError(mask,PX_FG(:,:,:,1),PX_BG(:,:,:,2));
error_EM(3,:) = countError(mask,PX_FG(:,:,:,1),PX_BG(:,:,:,3));
error_EM(4,:) = countError(mask,PX_FG(:,:,:,1),PX_BG(:,:,:,4));
error_EM(5,:) = countError(mask,PX_FG(:,:,:,1),PX_BG(:,:,:,5));

% Set 2
error_EM(6,:) = countError(mask,PX_FG(:,:,:,2),PX_BG(:,:,:,1));
error_EM(7,:) = countError(mask,PX_FG(:,:,:,2),PX_BG(:,:,:,2));
error_EM(8,:) = countError(mask,PX_FG(:,:,:,2),PX_BG(:,:,:,3));
error_EM(9,:) = countError(mask,PX_FG(:,:,:,2),PX_BG(:,:,:,4));
error_EM(10,:) = countError(mask,PX_FG(:,:,:,2),PX_BG(:,:,:,5));

% Set 3
error_EM(11,:) = countError(mask,PX_FG(:,:,:,3),PX_BG(:,:,:,1));
error_EM(12,:) = countError(mask,PX_FG(:,:,:,3),PX_BG(:,:,:,2));
error_EM(13,:) = countError(mask,PX_FG(:,:,:,3),PX_BG(:,:,:,3));
error_EM(14,:) = countError(mask,PX_FG(:,:,:,3),PX_BG(:,:,:,4));
error_EM(15,:) = countError(mask,PX_FG(:,:,:,3),PX_BG(:,:,:,5));

% Set 4
error_EM(16,:) = countError(mask,PX_FG(:,:,:,4),PX_BG(:,:,:,1));
error_EM(17,:) = countError(mask,PX_FG(:,:,:,4),PX_BG(:,:,:,2));
error_EM(18,:) = countError(mask,PX_FG(:,:,:,4),PX_BG(:,:,:,3));
error_EM(19,:) = countError(mask,PX_FG(:,:,:,4),PX_BG(:,:,:,4));
error_EM(20,:) = countError(mask,PX_FG(:,:,:,4),PX_BG(:,:,:,5));

% Set 5
error_EM(21,:) = countError(mask,PX_FG(:,:,:,5),PX_BG(:,:,:,1));
error_EM(22,:) = countError(mask,PX_FG(:,:,:,5),PX_BG(:,:,:,2));
error_EM(23,:) = countError(mask,PX_FG(:,:,:,5),PX_BG(:,:,:,3));
error_EM(24,:) = countError(mask,PX_FG(:,:,:,5),PX_BG(:,:,:,4));
error_EM(25,:) = countError(mask,PX_FG(:,:,:,5),PX_BG(:,:,:,5));

%%
% **************Generate the plot of the PoE vs. Dimensions***************
figure
hold on;
plot(dimensions,error_EM(1,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(2,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(3,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(4,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(5,:),'-o','MarkerSize',3,'LineWidth',1.5);
legend('BG1','BG2','BG3','BG4','BG5');
title('The PoE vs. Dimension: 1st FG Initialization');
xlabel('Dimensions');
ylabel('PoE')
grid on;
hold off;

figure(2)
hold on;
plot(dimensions,error_EM(6,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(7,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(8,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(9,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(10,:),'-o','MarkerSize',3,'LineWidth',1.5);
legend('BG1','BG2','BG3','BG4','BG5');
title('The PoE vs. Dimension: 2nd FG Initialization');
xlabel('Dimensions');
ylabel('PoE');
grid on;
hold off;

figure(3)
hold on;
plot(dimensions,error_EM(11,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(12,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(13,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(14,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(15,:),'-o','MarkerSize',3,'LineWidth',1.5);
legend('BG1','BG2','BG3','BG4','BG5');
title('The PoE vs. Dimension: 3rd FG Initialization');
xlabel('Dimensions');
ylabel('PoE');
grid on;
hold off;

figure(4)
hold on;
plot(dimensions,error_EM(16,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(17,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(18,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(19,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(20,:),'-o','MarkerSize',3,'LineWidth',1.5);
legend('BG1','BG2','BG3','BG4','BG5');
title('The PoE vs. Dimension: 4th FG Initialization');
xlabel('Dimensions');
ylabel('PoE');
grid on;
hold off;

figure(5)
hold on;
plot(dimensions,error_EM(21,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(22,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(23,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(24,:),'-o','MarkerSize',3,'LineWidth',1.5);
plot(dimensions,error_EM(25,:),'-o','MarkerSize',3,'LineWidth',1.5);
legend('BG1','BG2','BG3','BG4','BG5');
title('The PoE vs. Dimension: 5th FG Initialization');
xlabel('Dimensions');
ylabel('PoE');
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
    errs = zeros(1,11);
    pred = zeros(250,270);
    for j = 1:11
        count = 0;
        for x = 1:247
            for y = 1:262
                if p_bg(x,y,j)<p_fg(x,y,j)
                    pred(x,y) = 1;
                end

                if pred(x,y) ~= mask(x,y)
                    count = count + 1; 
                end
            end
        end
        errs(j) = count/(255*270);
    end

    figure
    imagesc(pred);
    colormap(gray(255));
end