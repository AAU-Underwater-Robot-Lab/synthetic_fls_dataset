load("breastTumorDeepLabv3-2025-01-23-13-17-52.mat",'transferTrainedNet');

%% Images
imds = imageDatastore("/media/seabrain/Expansion/marine-debris-fls-datasets-master/md_fls_dataset/data/watertank-segmentation/Images");

%% Labels
classNames = ["object","background"];
labelIDs = {[1; 2; 3; 5; 6; 7; 8; 9; 10; 11], [0]};
numClasses = numel(classNames);
pxds = pixelLabelDatastore("/media/seabrain/Expansion/marine-debris-fls-datasets-master/md_fls_dataset/data/watertank-segmentation/Masks",classNames,labelIDs,IncludeSubfolders=true);

%% Preview augmentation and mask
figure(2);

testImage = imds.preview();
mask = pxds.preview();
B = labeloverlay(testImage,mask,Transparency=0.7, ...
    IncludedLabels="object", ...
    Colormap="hsv");
imshowpair(testImage,int8(mask),"montage","Interpolation","bilinear");
title("Sonar Image & Labeled Training Example")

%% Test it!
pxdsResults = semanticseg(imds,transferTrainedNet,Verbose=true,Classes=classNames,OutputFolderName="segoutput_posttrain_real");
disp("After training: ")
disp("---------------------------")
metrics = evaluateSemanticSegmentation(pxdsResults,pxds,Verbose=true);

%display_result('output_example_posttrain_real',imds.preview, pxdsResults.preview, ranges, azimuths, {0, 1; 'object', 'background'});
%display_result('output_example_gt_real',imds.preview, pxds.preview, ranges, azimuths, {0, 1; 'object', 'background'});

%%
% Display example
overlayImg = labeloverlay(imds.readimage(10),pxdsResults.readimage(10),Transparency=0.7,...
    Colormap="hsv");
imshowpair(imds.readimage(10),overlayImg,"montage");
exportgraphics(gcf,'output_example_real_posttrain.png','Resolution',600)
exportgraphics(gcf,'output_example_real_posttrain.eps')

overlayImg = labeloverlay(imds.readimage(10),pxds.readimage(10),Transparency=0.7,...
    Colormap="hsv");
imshowpair(imds.readimage(10),overlayImg,"montage");
exportgraphics(gcf,'output_example_real_gt.png','Resolution',600)
exportgraphics(gcf,'output_example_real_gt.eps')

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
