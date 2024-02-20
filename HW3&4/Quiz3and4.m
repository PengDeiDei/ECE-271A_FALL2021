% *********Strategy 1*********
load('TrainingSamplesDCT_subsets_8.mat');
load("Alpha.mat");
%load("Prior_1.mat");
load("Prior_2.mat");
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
%%
% mean_FG = mean(D1_FG);
% mean_BG = mean(D1_BG);
% cov_FG = cov(D1_FG);
% cov_BG = cov(D1_BG);
% len_FG = length(D1_FG);
% len_BG = length(D1_BG);
% PY_FG = length(D1_FG)/(length(D1_FG)+length(D1_BG));
% PY_BG = length(D1_BG)/(length(D1_FG)+length(D1_BG));


% mean_FG = mean(D2_FG);
% mean_BG = mean(D2_BG);
% cov_FG = cov(D2_FG);
% cov_BG = cov(D2_BG);
% len_FG = length(D2_FG);
% len_BG = length(D2_BG);
% PY_FG = length(D2_FG)/(length(D2_FG)+length(D2_BG));
% PY_BG = length(D2_BG)/(length(D2_FG)+length(D2_BG));

% mean_FG = mean(D3_FG);
% mean_BG = mean(D3_BG);
% cov_FG = cov(D3_FG);
% cov_BG = cov(D3_BG);
% len_FG = length(D3_FG);
% len_BG = length(D3_BG);
% PY_FG = length(D3_FG)/(length(D3_FG)+length(D3_BG));
% PY_BG = length(D3_BG)/(length(D3_FG)+length(D3_BG));

mean_FG = mean(D4_FG);
mean_BG = mean(D4_BG);
cov_FG = cov(D4_FG);
cov_BG = cov(D4_BG);
len_FG = length(D4_FG);
len_BG = length(D4_BG);
PY_FG = length(D4_FG)/(length(D4_FG)+length(D4_BG));
PY_BG = length(D4_BG)/(length(D4_FG)+length(D4_BG));

errorPD = zeros(1,9); % Error of Predictive Distribution
errorML = zeros(1,9); % Error of Maximum Likehood
errorMAP = zeros(1,9); % Error of Maximun Per
%% 
% Predictive Distribution
for a = 1:length(alpha)
    sigma_0 = diag(alpha(a)*W0);

    part1_FG = (len_FG*sigma_0/(cov_FG+len_FG*sigma_0))*mean_FG';
    part2_FG = (cov_FG/(cov_FG+len_FG*sigma_0))*mu0_FG';
    mu_n_FG = part1_FG+part2_FG;
    sigma_n_FG = (cov_FG*sigma_0)/(cov_FG+len_FG*sigma_0);
    sigma_n_FG_Comb = sigma_n_FG+cov_FG;
    
    part1_BG = (len_BG*sigma_0/(cov_BG+len_BG*sigma_0))*mean_BG';
    part2_BG = (cov_BG/(cov_BG+len_BG*sigma_0))*mu0_BG';
    mu_n_BG = part1_BG+part2_BG;
    sigma_n_BG = (cov_BG*sigma_0)/(cov_BG+len_BG*sigma_0);
    sigma_n_BG_Comb = sigma_n_BG+cov_BG;

    % BDR
    img_BDR = zeros([r,m]);
    X = zeros([1,64]);
    count = 0;
    for i = 1:row-8
        for j = 1:colm-8
            X(1,:) = img_zzs(i,j,:);
            PX_T_FG = log(sqrt((2*pi)^64*det(sigma_n_FG_Comb))^(-1)*exp(-(X-mu_n_FG')/sigma_n_FG_Comb*(X-mu_n_FG')'/2)*PY_FG);
            PX_T_BG = log(sqrt((2*pi)^64*det(sigma_n_BG_Comb))^(-1)*exp(-(X-mu_n_BG')/sigma_n_BG_Comb*(X-mu_n_BG')'/2)*PY_BG);
    
            if PX_T_FG > PX_T_BG
               img_BDR(i,j) = 1; 
            end
    
            if mask(i,j) ~= img_BDR(i,j)
                count = count+1;
            end
        end
    end
%     figure(1)
%     subplot(3,3,a)
%     imagesc(img_BDR);
%     colormap(gray(255));

    errorPD(a) = count/(row*colm);
end
% Mximum Likehood
for a = 1:length(alpha)
    img_ML = zeros([r,m]);
    X = zeros([1,64]);
    count = 0;
    for i = 1:row-8
        for j = 1:colm-8
            X(1,:) = img_zzs(i,j,:);
            PX_FG = log(sqrt((2*pi)^64*det(cov_FG))^(-1)*exp(-(X-mean_FG)/cov_FG*(X-mean_FG)'/2)*PY_FG);
            PX_BG = log(sqrt((2*pi)^64*det(cov_BG))^(-1)*exp(-(X-mean_BG)/cov_BG*(X-mean_BG)'/2)*PY_BG);
    
            if PX_FG > PX_BG
               img_ML(i,j) = 1; 
            end
    
            if mask(i,j) ~= img_ML(i,j)
                count = count+1;
            end
        end
    end
%     figure(2)
%     subplot(3,3,a)
%     imagesc(img_ML);
%     colormap(gray(255));

    errorML(a) = count/(row*colm);
end
% Maximum a posteriori
for a = 1:length(alpha)
    sigma_0 = diag(alpha(a)*W0);

    part1_FG = (len_FG*sigma_0/(cov_FG+len_FG*sigma_0))*mean_FG';
    part2_FG = (cov_FG/(cov_FG+len_FG*sigma_0))*mu0_FG';
    mu_n_FG = part1_FG+part2_FG;
    
    part1_BG = (len_BG*sigma_0/(cov_BG+len_BG*sigma_0))*mean_BG';
    part2_BG = (cov_BG/(cov_BG+len_BG*sigma_0))*mu0_BG';
    mu_n_BG = part1_BG+part2_BG;

    % BDR
    img_MAP = zeros([r,m]);
    X = zeros([1,64]);
    count = 0;
    for i = 1:row-8
        for j = 1:colm-8
            X(1,:) = img_zzs(i,j,:);
            PX_FG_MAP = log(sqrt((2*pi)^64*det(cov_FG))^(-1)*exp(-(X-mu_n_FG')/cov_FG*(X-mu_n_FG')'/2)*PY_FG);
            PX_BG_MAP = log(sqrt((2*pi)^64*det(cov_BG))^(-1)*exp(-(X-mu_n_BG')/cov_BG*(X-mu_n_BG')'/2)*PY_BG);
    
            if PX_FG_MAP > PX_BG_MAP
               img_MAP(i,j) = 1; 
            end
    
            if mask(i,j) ~= img_MAP(i,j)
                count = count+1;
            end
        end
    end
%     figure(3)
%     subplot(3,3,a)
%     imagesc(img_MAP);
%     colormap(gray(255));

    errorMAP(a) = count/(row*colm);
end

%
figure(1)
hold on;
plot(alpha,errorPD);
plot(alpha,errorML);
plot(alpha,errorMAP);
hold off;
set(gca,'XScale','log');
legend('PD','ML','MAP')
title('Probability Error of Three Predictions Methods for D4,Strategy 2');

%%
figure(2)
subplot(2,2,1)
imagesc(img_BDR);
title('BDR');
colormap(gray(255));
subplot(2,2,2)
imagesc(img_ML);
title('ML');
colormap(gray(255));
subplot(2,2,3)
imagesc(img_MAP);
title('MAP');
colormap(gray(255));