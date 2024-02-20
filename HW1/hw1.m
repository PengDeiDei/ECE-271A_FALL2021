%% a)
samples= load('TrainingSamplesDCT_8.mat');

BG = samples.TrainsampleDCT_BG;
FG = samples.TrainsampleDCT_FG;

BGsize = size(BG,1) * size(BG,2); % number of samples in the set of background
FGsize = size(FG,1) * size(FG,2); % number of samples in the set of foreground 

Ysize = BGsize+ FGsize; % number of samples in the total training set 
Pyc = FGsize / Ysize; % P_Y(Cheetah) 
Pyg = BGsize / Ysize; % P_Y(Grass)

%% b) 
Xbg = zeros([1 64]);
Xfg = zeros([1 64]);

for i = 1:size(BG,1)
    temp = sort(BG(i,:),'descend');
    Xbg(BG(i,:)==temp(2)) = Xbg(BG(i,:)==temp(2)) + 1;
end

for i = 1:size(FG,1)
    temp = sort(FG(i,:),'descend');
    Xfg(FG(i,:)==temp(2)) = Xfg(FG(i,:)==temp(2)) + 1;
end

Pxyg = Xbg/size(BG,1); % P_X|Y(x|grass)
Pxyc = Xfg/size(FG,1); % P_X|Y(x|cheetah)

figure
subplot(1,2,1);
bar(Pxyg);
title('Index Histogram of P_{X|Y}(x|grass)');

subplot(1,2,2);
bar(Pxyc);
title('Index Histogram of P_{X|Y}(x|cheetah)');

%% c)
img= im2double(imread('cheetah.bmp'));
[row, colm] = size(img);

blocks = zeros(row-8,colm-8);
A = zeros(row-8,colm-8);
%read Zig-Zag Pattern.txt file
ZigZag = fopen('Zig-Zag Pattern.txt','r');
zzPat = fscanf(ZigZag,'%d',[8,8]);
fclose(ZigZag);

for i = 1:row-8
    for j = 1:colm-8
        dctImg = dct2(img(i:i+7,j:j+7));

        zzScan= zeros([1, 64]);
        for x = 1:8
            for y = 1:8
                zzScan(zzPat(x,y)+1) = abs(dctImg(x,y)); 
            end
        end
        
        tempZZ = sort(zzScan,'descend');
        blocks(i,j) = find(zzScan==tempZZ(2));
    end
end

%%
for i = 1:row-8
    for j = 1:colm-8
        if Pxyc(blocks(i,j))*Pyc >= Pxyg(blocks(i,j))*Pyg
            A(i,j) = 1;
        end
    end
end

figure
imagesc(A);
colormap(gray(255));
title(['The Predition of ','cheetah.bmp']);

%%
ground_truth = im2double(imread('cheetah_mask.bmp'));

% Padding to make the predition image the same size as the mask image
% The size of predition image is 247 x 262
PredImg = padarray(A, [4,4], 0);

missFG = 0;
missBG = 0;
gtFG = 0;
gtBG = 0;

for i = 1:size(ground_truth,1)
    for j = 1:size(ground_truth,2)
        if ground_truth(i,j) == 1
            gtFG = gtFG + 1;
            if PredImg(i,j) ~= ground_truth(i,j)
                missFG = missFG + 1;
            end
        else
            gtBG = gtBG + 1;
            if PredImg(i,j) ~= ground_truth(i,j)
                missBG = missBG + 1;
            end
        end
    end
end

% Calculate error
errFG = missFG / gtFG * Pyc;
errBG = missBG / gtBG * Pyg;
err = errFG + errBG;