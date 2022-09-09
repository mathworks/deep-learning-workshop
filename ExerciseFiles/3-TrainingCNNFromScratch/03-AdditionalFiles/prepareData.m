function [imgDataTrain, labelsTrain, imgDataTest, labelsTest] = prepareData

%% Check for the existence of the MNIST files and download them if necessary

files = [   "train-images-idx3-ubyte",...
            "train-labels-idx1-ubyte",...
            "t10k-images-idx3-ubyte",...
            "t10k-labels-idx1-ubyte"  ];

% boolean for testing if the files exist
% basically, check for existence of "data" directory
download = exist(fullfile(pwd, 'data'), 'dir') ~= 7;

if download
    disp('Downloading files...')
    mkdir data
    webPrefix = "http://yann.lecun.com/exdb/mnist/";
    webSuffix = ".gz";

    filenames = files + webSuffix;
    for ii = 1:numel(files)
        websave(fullfile('data', filenames{ii}),...
            char(webPrefix + filenames(ii)));
    end
    disp('Download complete.')
    
    % unzip the files
    cd data
    gunzip *.gz
    
    % return to main directory
    cd ..
end

%% Extract the MNIST images into arrays

disp('Preparing MNIST data...');

% Read headers for training set image file
fid = fopen(fullfile('data/', char(files{1})), 'r', 'b');
magicNum = fread(fid,1,'int32');
numImgs  = fread(fid, 1, 'uint32');
numRows  = fread(fid, 1, 'uint32');
numCols  = fread(fid, 1, 'uint32');

% Read the data part 
rawImgDataTrain = uint8(fread(fid, numImgs * numRows * numCols, 'uint8'));
fclose(fid);

% Reprocess the data part into a 4D array
rawImgDataTrain = reshape(rawImgDataTrain, [numRows, numCols, numImgs]);
rawImgDataTrain = permute(rawImgDataTrain, [2,1,3]);
imgDataTrain = uint8(zeros(numRows, numCols, 1, numImgs));
for ii = 1:numImgs
    imgDataTrain(:,:,1,ii) = uint8(rawImgDataTrain(:,:,ii));
end

% Read headers for training set label file
fid = fopen(fullfile('data/', char(files{2})), 'r', 'b');
magicNum = fread(fid,1,'int32');
numLabels = fread(fid, 1, 'uint32');

% Read the data for the labels
labelsTrain = fread(fid, numLabels, 'uint8');
fclose(fid);

% Process the labels
labelsTrain = categorical(labelsTrain);

% Read headers for test set image file
fid = fopen(fullfile('data/', char(files{3})), 'r', 'b');
magicNum = fread(fid,1,'int32');
numImgs  = fread(fid, 1, 'uint32');
numRows  = fread(fid, 1, 'uint32');
numCols  = fread(fid, 1, 'uint32');

% Read the data part 
rawImgDataTest = uint8(fread(fid, numImgs * numRows * numCols, 'uint8'));
fclose(fid);

% Reprocess the data part into a 4D array
rawImgDataTest = reshape(rawImgDataTest, [numRows, numCols, numImgs]);
rawImgDataTest = permute(rawImgDataTest, [2,1,3]);
imgDataTest = uint8(zeros(numRows, numCols, 1, numImgs));
for ii = 1:numImgs
    imgDataTest(:,:,1,ii) = uint8(rawImgDataTest(:,:,ii));
end

% Read headers for test set label file
fid = fopen(fullfile('data/', char(files{4})), 'r', 'b');
magicNum = fread(fid,1,'int32');
numLabels = fread(fid, 1, 'uint32');

% Read the data for the labels
labelsTest = fread(fid, numLabels, 'uint8');
fclose(fid);

% Process the labels
labelsTest = categorical(labelsTest);

disp('MNIST data preparation complete.');