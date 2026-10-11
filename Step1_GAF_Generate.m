% 清除工作区和命令行窗口
clc
clear 
close all;
% load iris.mat;
Iris = xlsread('数据1.xlsx');
data = Iris(:,2:end); %这里只选取特征列

% 获取数据集的样本数量和每个样本的长度
[numSamples, sampleLength] = size(data);

% 循环处理每个样本数据
for imgIdx = 1:numSamples
   
    %% 生成数据
    % 从data中获取当前样本数据
    speedData = data(imgIdx,:);

    % 对样本数据进行归一化到[0, 1]范围
    normalizedData = (speedData - min(speedData)) / (max(speedData) - min(speedData));

    %% 数据处理
    % 将归一化后的数据映射到[-1, 1]范围
    scaledData = ((normalizedData - max(normalizedData)) + (normalizedData - min(normalizedData))) / (max(normalizedData) + min(normalizedData));

    % 求极坐标下的角度
    polarAngle = acos(scaledData);

    % 生成格拉姆角场矩阵
    GASF = scaledData' * scaledData - sqrt(1 - scaledData.^2)' * sqrt(1 - scaledData.^2);
    GADF = sqrt(1 - scaledData.^2)' * scaledData + scaledData' * sqrt(1 - scaledData.^2);

    %% 生成图像文件
    % 创建图像文件名
    imgFileName = [num2str(imgIdx), '.png'];

    % 保存 GSAF 图像
    im = figure('Visible','off');
    imagesc(GASF);
    colormap('jet'); % 使用 jet 颜色映射
    set(gca, 'looseInset', [0 0 0 0]); % 去除图像边框
    axis off 
    set(gca, 'xtick', [], 'ytick', [], 'xcolor', 'w', 'ycolor', 'w'); % 隐藏坐标轴
    % 保存图像文件
    saveas(im, fullfile('GASF图像', imgFileName));
    close all; % 关闭图像窗口

    
    % 保存 GSDF 图像
    im = figure('Visible','off');
    imagesc(GADF);
    colormap('jet'); % 使用 jet 颜色映射
    set(gca, 'looseInset', [0 0 0 0]); % 去除图像边框
    axis off 
    set(gca, 'xtick', [], 'ytick', [], 'xcolor', 'w', 'ycolor', 'w'); % 隐藏坐标轴
    % 保存图像文件
    saveas(im, fullfile('GADF图像', imgFileName));
    close all; % 关闭图像窗口

end
