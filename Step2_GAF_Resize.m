clc
clear all
% load iris.mat 
Iris = xlsread('数据1.xlsx');%这里加载原数据，只是为了获取标签列
label = Iris(:,1);  %取出标签列

inputFolder1 = 'GASF图像'; % 修改为实际的输入文件夹路径（未改分辨率）
inputFolder2 = 'GADF图像'; % 修改为实际的输入文件夹路径（未改分辨率）

% 获取文件夹中所有的图像文件
imageFiles = dir(fullfile(inputFolder1, '*.png'));

% 提取文件名中的数字部分
imageNumbers = cellfun(@(x) sscanf(x, '%d.png'), {imageFiles.name});

% 按照数字部分排序
[sortedNumbers, sortOrder] = sort(imageNumbers);

% 根据排序后的索引处理图像文件
sortedImageFiles = imageFiles(sortOrder);
numImages = length(sortedImageFiles);

imageSize = [64,NaN,3]; % 目标图像尺寸

for i = 1:numImages
    i
    imageFileName = sortedImageFiles(i).name;
    imagePath = fullfile(inputFolder1, imageFileName);
    % 读取GASF图像
    img = imread(imagePath);
    % 调整图像尺寸
    imgResized = imresize(img, imageSize(1:2)); 
    imgg = double(imgResized)/255;
    GASF{1,i} = imgg;
    GASF{2,i} = label(i);

    imageFileName = sortedImageFiles(i).name;
    imagePath = fullfile(inputFolder2, imageFileName);
    % 读取GADF图像
    img = imread(imagePath);
    % 调整图像尺寸
    imgResized = imresize(img, imageSize(1:2)); 
    imgg = double(imgResized)/255;
    GADF{1,i} = imgg;
    GADF{2,i} = label(i);
end
save 'GAF.mat' GASF GADF
disp('处理完成！');
