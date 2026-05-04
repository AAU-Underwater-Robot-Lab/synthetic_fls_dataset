load("breast_seg_deepLabV3_v2.mat",'trainedNet');

%% Pretrained network example
%figure(1)
imTest = imread("breastUltrasoundImg.png");
%imSize = [256 256];
%imTest = imresize(imTest,imSize);

classNames = ["object","background"];
segmentedImg = semanticseg(imTest,trainedNet,Classes=classNames,OutputFolderName="segoutput_pretrain");

overlayImg = labeloverlay(imTest,segmentedImg,Transparency=0.7,IncludedLabels="object", ...
    Colormap="hsv");
imshowpair(imTest,overlayImg,"montage");
exportgraphics(gcf,'output_example_sonogram.png','Resolution',600)
exportgraphics(gcf,'output_example_sonogram.eps')

%% Images
imds = imageDatastore("output");
imds = subset(imds,contains(imds.Files,"_sonar.png"));

%% Labels
labelIDs = {[1; 2; 3; 5; 6; 7], [0;4]};
numClasses = numel(classNames);
pxds = pixelLabelDatastore("output",classNames,labelIDs,IncludeSubfolders=true);
pxds = subset(pxds,contains(pxds.Files,"_segmap.png"));


%% Split dataset
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds, pxds);
dsTrain = combine(imdsTrain, pxdsTrain);
dsVal = combine(imdsVal, pxdsVal);
dsTest = combine(imdsTest, pxdsTest);

%% Augmentation
tdsTrain = transform(dsTrain,@transformBreastTumorImageAndLabels,IncludeInfo=true);
tdsVal = transform(dsVal,@transformBreastTumorImageAndLabels,IncludeInfo=true);
tdsTest = transform(dsTest,@transformBreastTumorImageResize,IncludeInfo=true);

%% Preview augmentation and mask
figure(2);
testImageFull = dsTrain.preview{1};
testImage = tdsTrain.preview{1};
mask = tdsTrain.preview{2};
B = labeloverlay(testImage,mask,Transparency=0.7, ...
    IncludedLabels="object", ...
    Colormap="hsv");
imshowpair(testImageFull,imresize(B,"nearest","OutputSize",size(testImageFull)),"montage","Interpolation","bilinear");
title("Sonar Image & Labeled Training Example")
%% Options
options = trainingOptions("adam", ...
    ExecutionEnvironment="gpu", ...
    InitialLearnRate=1e-3, ...
    ValidationData=tdsVal, ...
    MaxEpochs=100, ...
    MiniBatchSize=2, ...
    VerboseFrequency=20, ...
    ValidationFrequency=10,...
    ValidationPatience=10,...
    OutputNetwork="best-validation-loss",...
    Plots="training-progress");

%% Network setup
imageSize = [256 256 3];
untrainedNet = deeplabv3plusLayers(imageSize,numClasses,"resnet50");
newInputLayer = imageInputLayer(imageSize(1:2),Name="newInputLayer");
untrainedNet = replaceLayer(untrainedNet,untrainedNet.Layers(1).Name,newInputLayer);
newConvLayer = convolution2dLayer([7 7],64,Stride=2,Padding=[3 3 3 3],Name="newConv1");
untrainedNet = replaceLayer(untrainedNet,untrainedNet.Layers(2).Name,newConvLayer);

%% Class imbalance pixel layer
tbl = countEachLabel(pxds);
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','classification','Classes',tbl.Name,'ClassWeights',classWeights);

%% Modify untrained network
untrainedNet = replaceLayer(untrainedNet,"classification",pxLayer);

%% Test before training
pretrainedNet = trainedNet; % From file!
pxdsResults = semanticseg(tdsTest,pretrainedNet,Verbose=true,Classes=classNames);
disp("Before training: ")
disp("---------------------------")
metrics_before = evaluateSemanticSegmentation(pxdsResults,tdsTest,Verbose=true);

% Display example
display_result('output_example_pretrain',tdsTest.preview{1}, pxdsResults.preview, ranges, azimuths, {0, 1; 'object', 'background'});

%overlayImg = labeloverlay(tdsTest.preview{1},pxdsResults.preview,Transparency=0.7,...
%    Colormap="hsv");
%imshowpair(tdsTest.preview{1},overlayImg,"montage");
%exportgraphics(gcf,'output_example_pretrain.png','Resolution',600)
%exportgraphics(gcf,'output_example_pretrain.eps')

%% Modify externally pertrained sonogram net
pretrainedNet = addLayers(pretrainedNet.layerGraph,pxLayer);
pretrainedNet = connectLayers(pretrainedNet,'softmax-out','classification/in');

%% Training
doTraining = true;
if doTraining
    transferTrainedNet = trainNetwork(tdsTrain,pretrainedNet,options);
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save("breastTumorDeepLabv3-"+modelDateTime+".mat","transferTrainedNet");
end

%% Test it!
pxdsResults = semanticseg(tdsTest,transferTrainedNet,Verbose=true,Classes=classNames,OutputFolderName="segoutput_posttrain");
disp("After training: ")
disp("---------------------------")
metrics = evaluateSemanticSegmentation(pxdsResults,tdsTest,Verbose=true);

display_result('output_example_posttrain',tdsTest.preview{1}, pxdsResults.preview, ranges, azimuths, {0, 1; 'object', 'background'});

% Display example
%overlayImg = labeloverlay(tdsTest.preview{1},pxdsResults.preview,Transparency=0.7,...
%    Colormap="hsv");
%imshowpair(tdsTest.preview{1},overlayImg,"montage");
%exportgraphics(gcf,'output_example_posttrain.png','Resolution',600)
%exportgraphics(gcf,'output_example_posttrain.eps')

%% Helpers
function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds)
% Partition CamVid data by randomly selecting 60% of the data for training. The
% rest is used for testing.
% Set initial random state for example reproducibility.
    rng(0); 
    numFiles = numel(imds.Files);
    shuffledIndices = randperm(numFiles);
    % Use 80% of the images for training.
    numTrain = round(0.80 * numFiles);
    trainingIdx = shuffledIndices(1:numTrain);
    % Use 10% of the images for validation
    numVal = round(0.10 * numFiles);
    valIdx = shuffledIndices(numTrain+1:numTrain+numVal);
    % Use the rest for testing.
    testIdx = shuffledIndices(numTrain+numVal+1:end);
    % Create image datastores for training and test.
    imdsTrain = subset(imds,trainingIdx);
    imdsVal = subset(imds,valIdx);
    imdsTest = subset(imds,testIdx);
    % Create pixel label datastores for training and test.
    pxdsTrain = subset(pxds,trainingIdx);
    pxdsVal = subset(pxds,valIdx);
    pxdsTest = subset(pxds,testIdx);
end

function display_result(fname, intensityGrid, segmapGrid, ranges, azimuths, classes)
    %% Display
    fig(41);
    clf;
    densityGrid = intensityGrid;
    densityGrid(isnan(densityGrid)) = 0;
    nclasses = numel(classes)/2;

    size_im = [numel(ranges), numel(azimuths)];
    % Polar plot for intensity grid
    h1 = polarPcolor(ranges, rad2deg(azimuths), im2double(imresize(densityGrid,size_im)), 'Nspokes',7,'autoOrigin','off','colbar',0); % Use the downloaded polarPcolor function
    
    set(gcf, 'Name', 'Sonar 2D with Intensity in Polar Coordinates');
    xlabel('Azimuth');
    ylabel('Range');
    cb1 = colorbar(gca,"westoutside");
    clim([0 1]);
    cb1.Label.String = 'Intensity (Normalized)';
    colormap('hot');

    set(gcf, 'Position', [100, 100, 700, 500]);
    drawnow();

    exportgraphics(gcf, strcat(fname, '_sonarplot_polar.png'), 'Resolution', 600);
    exportgraphics(gcf, strcat(fname, '_sonarplot_polar.eps'));
    
    % Plot segmap grid in polar coordinates
    fig(42);
    clf;
    h2 = polarPcolor(ranges, rad2deg(azimuths), imresize(segmapGrid,size_im),'Nspokes',7,'autoOrigin','off','ncolor',nclasses,'colormap','turbo','colbar',0); % Use the downloaded polarPcolor function
    set(gcf, 'Name', 'Sonar segmap in Polar Coordinates');
    xlabel('Azimuth');
    ylabel('Range');
    
    %colormap(colorss);
    cb2 = colorbar(gca,"westoutside");
    cb2.Label.String = 'Class';
    
    ticks = linspace(cb2.Limits(1), cb2.Limits(2), 1 + 2 * nclasses);
    ticks(1:2:end) = [];
    labels([classes{1,:}]+1) = classes(2,:);
    cb2.Ticks = ticks;
    cb2.TickLabels = labels;
    
    % createlegend(colorss, )
    set(gcf, 'Position', [100, 600, 700, 500]);
    drawnow();

    exportgraphics(gcf, strcat(fname, '_labelplot_polar.png'), 'Resolution', 300);
    exportgraphics(gcf, strcat(fname, '_labelplot_polar.eps'));

    
end
