%% 初始化
clear
close all
clc
warning off

%% 数据读取
addpath(genpath(pwd));
load GAF.mat                                      % 加载数据

for i = 1:size(GADF,2)   
    input1(:,:,:,i) = GADF{1,i};  
end
for i = 1:size(GASF,2) 
    input2(:,:,:,i) = GASF{1,i};  
end
output=GADF(2,:);

%划分训练集与测试集
num_samples = size(GASF,2) ;                      % 样本个数
idx =randperm(num_samples);
num_size = 0.7;                                   % 训练集占数据集比例
num_train_s = round(num_size * num_samples);      % 训练集样本个数
train_X1 = input1(:,:,:,idx(1:num_train_s));
train_X2 = input2(:,:,:,idx(1:num_train_s));
train_Y = output(:,idx(1:num_train_s))';

test_X1 = input1(:,:,:,idx(num_train_s+1:num_samples));
test_X2 = input2(:,:,:,idx(num_train_s+1:num_samples));
test_Y = output(:,idx(num_train_s+1:num_samples))';

train_Y = categorical(cell2mat(train_Y));
test_Y =  categorical(cell2mat(test_Y));

dsX1Train = arrayDatastore(train_X1,IterationDimension=4);
dsX2Train = arrayDatastore(train_X2,IterationDimension=4);
dsTTrain = arrayDatastore(train_Y);
dsTrain = combine(dsX1Train,dsX2Train,dsTTrain);  % 合并两个训练集输入和标签

dsX1Test = arrayDatastore(test_X1,IterationDimension=4);
dsX2Test = arrayDatastore(test_X2,IterationDimension=4);
dsTTest = arrayDatastore(test_Y); 
dsTest = combine(dsX1Test,dsX2Test,dsTTest);      % 合并两个测试集输入和标签

classnum =3;                                    % 故障类别
numFeatures = size(input1,1);

%% 网络结构
layers1 = [
    imageInputLayer([size(input1,1) size(input1,2) size(input1,3)],'Name','input1')  
    
    convolution2dLayer([5 5],10,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer([2 2],'Stride',2,'Padding','same','name','maxpool1')   % 第一层池化层，包括3x3大小的池化窗口，步长为1，same填充方式
    dropoutLayer(0.2)

    flattenLayer('name','flatten1')

    fullyConnectedLayer(10,'Name','fc11')

    additionLayer(2,'Name','add')

    fullyConnectedLayer(classnum,'Name','fc12')

    
    selfAttentionLayer(1,25)                                                 % 创建一个单头，25个键和查询通道的自注意力层 
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')];

lgraph = layerGraph(layers1);

layers2 = [imageInputLayer([size(input2,1) size(input2,2) size(input2,3)],'Name','input2')  
    convolution2dLayer([5 5],10,'Padding','same','Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer([2 2],'Stride',2,'Padding','same','name','maxpool2')   % 第一层池化层，包括3x3大小的池化窗口，步长为1，same填充方式
    dropoutLayer(0.2)

    flattenLayer('name','flatten2')
    fullyConnectedLayer(10,'Name','fc21')];
lgraph = addLayers(lgraph,layers2);
lgraph = connectLayers(lgraph,'fc21','add/in2');

%% 设置参数
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 150, ...                             % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.001, ...                    % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod',100, ...                    % 训练100次后开始调整学习率
    'LearnRateDropFactor',0.01, ...                   % 学习率调整因子
    'L2Regularization', 0.001, ...                    % 正则化参数
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 1, ...                                 % 关闭优化过程
    'Plots', 'none');                                 % 画出曲线

%% 开始训练
t0 = tic;                                             % 开始计时
net = trainNetwork(dsTrain, lgraph,options);
toc(t0);                                              % 从t0开始到此处的执行时间
analyzeNetwork(net)

%% 准确率计算
pred = classify(net, dsTest);
accuracy=sum(test_Y==pred)/length(pred);              % 计算预测的确率

%% 画方框图
figure
confMat = confusionmat(test_Y,pred);
plot2(confMat.');
xlabel('Predicted label')
ylabel('Real label')
title(['并行CNN-Attention的测试集正确率 = ',num2str(accuracy*100),' %'])

%% 作图
figure
scatter(1:length(pred),pred,'r^')
hold on
scatter(1:length(pred),test_Y,'b*')
legend('预测类别','真实类别','NorthWest')
title(['并行CNN-Attention的测试集正确率 = ',num2str(accuracy*100),' %'])
xlabel('预测样本编号')
ylabel('分类结果')
box on
set(gca,'fontsize',12)

%% 原始样本分布
layerT = 'input2';
LayersTNeed = activations(net,dsTrain,layerT,'OutputAs','channels');
RawFeature = [];
for i = 1:size(LayersTNeed,4)
    temp = reshape(LayersTNeed(:,:,:,i),1,[]);
    RawFeature = [RawFeature;temp];
end
tsne_data = tsne(RawFeature);
temp = [];
NumTypes = classnum;                                   % 故障类别数
for i = 1:size(train_Y,1)
headers = {'直行';'右换道';'左换道'};
cmap = hsv(NumTypes);
lable = double(train_Y);

%% 计算训练集多少种故障
if lable(i)== 1
    str = headers(1);
elseif lable(i)== 2
    str = headers(2);
elseif lable(i)== 3
    str = headers(3);
end
tempdata = categorical(cellstr(str));
temp = [temp;tempdata];
end

%% 类别标签
species = temp;
figure 
gscatter(tsne_data(:,1),tsne_data(:,2),species,cmap,'.',20,'on');
sgtitle('原始样本分布', 'Interpreter', 'none', 'FontSize', 14);

%% 模型识别后样本分布
layerT = 'fc12';
LayersTNeed = activations(net,dsTrain,layerT,'OutputAs','channels');
RawFeature = [];
for i = 1:size(LayersTNeed,2)
    temp = reshape(LayersTNeed(:,i),1,[]);
    RawFeature = [RawFeature;temp];
end
tsne_data = tsne(RawFeature);
temp = [];
NumTypes = classnum;
for i = 1:size(train_Y,1)
headers = {'直行';'右换道';'左换道'};
cmap = hsv(NumTypes);
lable = double(train_Y);

%% 计算训练集多少种故障
if lable(i)== 1
    str = headers(1);
elseif lable(i)== 2
    str = headers(2);
elseif lable(i)== 3
    str = headers(3);
end
tempdata = categorical(cellstr(str));
temp = [temp;tempdata];
end
species = temp;
figure 
gscatter(tsne_data(:,1),tsne_data(:,2),species,cmap,'.',20,'on');
sgtitle('本文模型识别后的样本分布', 'Interpreter', 'none', 'FontSize', 14);