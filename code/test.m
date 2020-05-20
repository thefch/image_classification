function test(train_image_paths,test_image_paths,train_labels,categories,test_labels)
    FEATURE ="bag of sift";%spatial pyramid,bag of sift
    switch(FEATURE)
        case "bag of sift"
            test_bag_of_sift(train_image_paths,test_image_paths,train_labels,categories,test_labels)
        case "spatial pyramid"
            test_spatial_pyramid(train_image_paths,test_image_paths,train_labels,categories,test_labels)
        case "fisher encoding"
            test_fisher_encoding(train_image_paths,test_image_paths,train_labels,categories,test_labels)
    end
end


function test_fisher_encoding(train_image_paths,test_image_paths,train_labels,categories,test_labels)
    FEATURE = "fisher encoding";%'colour histogram', 'tiny image','bag of sift', 'spatial pyramids'
   
    COLOUR_SPACE =["GRAYSCALE","RGB"];%,HSV, GRAYSCALE, YCBR, NTSC
    DISTANCE = 'L1'; %L1,chisq need double quotes
    STEP_SIZE = [10,100];
    BIN_SIZE=6;
    FEATURE_STEP_SIZE = 5;
    VOCAB_SIZE = [100,1000]; % you need to test the influence of this parameter
    CLASSIFIER = ["support vector machine"];%,'support vector machine';,,"nearest neighbor"

    fileID = fopen('results8fisher.txt','a');
    
    for ss=1:size(STEP_SIZE,2)
        fprintf("in progress :%d/%d  -- %d \n",ss,size(STEP_SIZE,2),STEP_SIZE(1,ss))
        for c=1:size(COLOUR_SPACE,2)
            for v=1:size(VOCAB_SIZE,2)
                fprintf("  --- voc:%d \n",VOCAB_SIZE(1,v));
                
                
                name_voc = sprintf('vocab_fisher_%d%s%d.mat',VOCAB_SIZE(1,v), COLOUR_SPACE(1,c),STEP_SIZE(1,ss));
                if ~exist(name_voc, 'file')
                    fprintf(' dictionary not found : %s \n',name_voc)
                    [means, covariances, priors] = build_vocab_fisher(train_image_paths, VOCAB_SIZE(1,v),STEP_SIZE(1,ss),COLOUR_SPACE(1,c));
                    save(name_voc, 'means', 'covariances', 'priors');
                 else
                    load(name_voc);
                end
                
                
                
                
     %image_paths,MAX_LEVEL,VOCAB_SIZE,DISTANCE,STEP_SIZE,COLORSPACE,BIN_SIZE,USE_PHOW,USE_NORM)
                name_feats = sprintf('feats_fisher_%d%s%d.mat',VOCAB_SIZE(1,v), COLOUR_SPACE(1,c),FEATURE_STEP_SIZE);
                if ~exist(name_feats, 'file')
                    fprintf(' CREATING FEATS: %s \n',name_feats)
                   
                    train_image_feats = fisher_encoding(train_image_paths,VOCAB_SIZE(1,v),FEATURE_STEP_SIZE,COLOUR_SPACE(1,c),BIN_SIZE,means, covariances, priors);
                    test_image_feats = fisher_encoding(test_image_paths,VOCAB_SIZE(1,v),FEATURE_STEP_SIZE,COLOUR_SPACE(1,c),BIN_SIZE,means, covariances, priors);
                    save(name_feats, 'train_image_feats', 'test_image_feats');
                else
                    load(name_feats);
                end    

                for class=1 :size(CLASSIFIER,2)
                   if(CLASSIFIER(1,class) == "support vector machine" ) 
                       lbs  = [10.0,1.0,0.1,0.01,0.001,0.0001,0.00001...
                           0.00011,0.00001,0.00002,0.00003,0.00004,0.00005,];
                       for lam=1:size(lbs,2)
                            accuracy = SVM(train_image_feats, train_labels, test_image_feats,categories,lbs(1,lam),test_labels);
                            output = sprintf("%s, %d, %d, %d, %s, %s, %f, %f \n",...
                                 FEATURE,VOCAB_SIZE(1,v),FEATURE_STEP_SIZE,STEP_SIZE(1,ss),COLOUR_SPACE(1,c),CLASSIFIER(1,class),lbs(1,lam),accuracy);
                            fprintf(fileID,output);
                       end
                   elseif(CLASSIFIER(1,class) == "nearest neighbor")
                       for nn=1:15
                            accuracy = kNN(nn,train_image_feats, train_labels, test_image_feats,categories,DISTANCE,test_labels);
                            output = sprintf("%s, %d, %d, %d, %s, %s, %d, %f \n",...
                                FEATURE,VOCAB_SIZE(1,v),FEATURE_STEP_SIZE,STEP_SIZE(1,ss),COLOUR_SPACE(1,c),CLASSIFIER(1,class),nn,accuracy);
                            fprintf(fileID,output);
                       end
                   end
                end
            end
        end
        fprintf("done :%d/%d \n",ss,size(STEP_SIZE,2))
    end  
end

function test_spatial_pyramid(train_image_paths,test_image_paths,train_labels,categories,test_labels)
    FEATURE = "spatial pyramid";%'colour histogram', 'tiny image','bag of sift', 'spatial pyramids'
   
    COLOUR_SPACE =["RGB"];%,HSV, GRAYSCALE, YCBR, NTSC
    DISTANCE = 'L1'; %L1,chisq need double quotes
    STEP_SIZE = [10,100];
    BIN_SIZE=6;
    FEATURE_STEP_SIZE = 5;
    VOCAB_SIZE = [100,1000]; % you need to test the influence of this parameter
    CLASSIFIER = ["support vector machine"];%,'support vector machine';
    %USE_PHOW = [false,true];
    USE_PHOW = true;
    USE_NORM = false;
    USE_MEAN = true;
    MAX_LEVEL=1;
    fileID = fopen('results7pyramidB.txt','a');
    
    for ss=1:size(STEP_SIZE,2)
        fprintf("in progress :%d/%d  -- %d \n",ss,size(STEP_SIZE,2),STEP_SIZE(1,ss))
        for c=1:size(COLOUR_SPACE,2)
            for v=1:size(VOCAB_SIZE,2)
                fprintf("  --- voc:%d \n",VOCAB_SIZE(1,v));
                name_voc = sprintf('vocab%d%s%d_phow%d.mat',VOCAB_SIZE(1,v), COLOUR_SPACE(1,c),STEP_SIZE(1,ss),USE_PHOW);
                if ~exist(name_voc, 'file')
                    fprintf(' dictionary not found : %s \n',name_voc)
                    vocab = build_vocabulary(train_image_paths, VOCAB_SIZE(1,v),STEP_SIZE(1,ss),COLOUR_SPACE(1,c),BIN_SIZE,USE_PHOW,USE_NORM);
                    save(name_voc, 'vocab');
                else
                    load(name_voc);
                end
     %image_paths,MAX_LEVEL,VOCAB_SIZE,DISTANCE,STEP_SIZE,COLORSPACE,BIN_SIZE,USE_PHOW,USE_NORM)
                name_feats = sprintf('featsSP%d%s%d_phow%d_level%d.mat',VOCAB_SIZE(1,v), COLOUR_SPACE(1,c),FEATURE_STEP_SIZE,USE_PHOW,MAX_LEVEL);
                if ~exist(name_feats, 'file')
                    fprintf(' CREATING FEATS: %s \n',name_feats)
                    
                    %spatial_pyramid(image_paths,MAX_LEVEL,VOCAB_SIZE,DISTANCE,STEP_SIZE,COLORSPACE,BIN_SIZE,USE_PHOW,USE_NORM)
                    train_image_feats   = spatial_pyramid(...
                        train_image_paths,MAX_LEVEL,VOCAB_SIZE(1,v),DISTANCE,FEATURE_STEP_SIZE,COLOUR_SPACE(1,c),BIN_SIZE,USE_PHOW,USE_NORM,vocab);
                    test_image_feats    = spatial_pyramid(...
                        test_image_paths,MAX_LEVEL,VOCAB_SIZE(1,v),DISTANCE,FEATURE_STEP_SIZE,COLOUR_SPACE(1,c),BIN_SIZE,USE_PHOW,USE_NORM,vocab);
                    save(name_feats, 'train_image_feats', 'test_image_feats');
                else
                    load(name_feats);
                end    

                for class=1 :size(CLASSIFIER,2)
                   if(CLASSIFIER(1,class) == "support vector machine" ) 
                       lbs  = [10.0,1.0,0.1,0.01,0.001,0.0001,0.00001...
                           0.00011,0.00001,0.00002,0.00003,0.00004,0.00005,];
                       for lam=1:size(lbs,2)
                            accuracy = SVM(train_image_feats, train_labels, test_image_feats,categories,lbs(1,lam),test_labels);
                            output = sprintf("%s, %d, %d, %d, %d, %s, %d, %s, %f, %f \n",...
                                FEATURE,VOCAB_SIZE(1,v),MAX_LEVEL,FEATURE_STEP_SIZE,STEP_SIZE(1,ss),COLOUR_SPACE(1,c),USE_PHOW,CLASSIFIER(1,class),lbs(1,lam),accuracy);
                            fprintf(fileID,output);
                       end
                   elseif(CLASSIFIER(1,class) == "nearest neighbor")
                       for nn=1:15
                            accuracy = kNN(nn,train_image_feats, train_labels, test_image_feats,categories,DISTANCE,test_labels);
                            output = sprintf("%s, %d, %d, %d, %d, %s, %d, %s, %d, %f \n",...
                                FEATURE,VOCAB_SIZE(1,v),MAX_LEVEL,FEATURE_STEP_SIZE,STEP_SIZE(1,ss),COLOUR_SPACE(1,c),USE_PHOW,CLASSIFIER(1,class),nn,accuracy);
                            fprintf(fileID,output);
                       end
                   end
                end
            end
        end
        fprintf("done :%d/%d \n",ss,size(STEP_SIZE,2))
    end  
end

function test_bag_of_sift(train_image_paths,test_image_paths,train_labels,categories,test_labels)
    FEATURE = "bag of sift";%'colour histogram', 'tiny image','bag of sift', 'spatial pyramids'
    
    COLOUR_SPACE =["RGB"];%,HSV, GRAYSCALE, YCBR, NTSC
    DISTANCE = 'L1'; %L1,chisq need double quotes
    STEP_SIZE = [10,100,200];
    BIN_SIZE=6;
    FEATURE_STEP_SIZE = 5;
    VOCAB_SIZE = [100,200,1000]; % you need to test the influence of this parameter
    CLASSIFIER = ["nearest neighbor"];%,'support vector machine';
    USE_PHOW = true;
    USE_NORM = false;
    USE_MEAN = true;
    fileID = fopen('results5siftB_phow1.txt','a');

    for v=1:size(VOCAB_SIZE,2)
        for c=1:size(COLOUR_SPACE,2)
            for ss=1:size(STEP_SIZE,2)
                name_voc = sprintf('vocab%d%s%d_phow%d.mat',VOCAB_SIZE(1,v), COLOUR_SPACE(1,c),STEP_SIZE(1,ss),USE_PHOW);
                if ~exist(name_voc, 'file')
                    fprintf(' dictionary not found : %s \n',name_voc)
                    vocab = build_vocabulary(train_image_paths, VOCAB_SIZE(1,v),STEP_SIZE(1,ss),COLOUR_SPACE(1,c),BIN_SIZE,USE_PHOW,USE_NORM);
                    save(name_voc, 'vocab');
                else
                    load(name_voc);
                end
                
                name_feats = sprintf('featsBoG%d%s%d_phow%d.mat',VOCAB_SIZE(1,v), COLOUR_SPACE(1,c),FEATURE_STEP_SIZE,USE_PHOW);
                if ~exist(name_feats, 'file')
                    fprintf(' CREATING FEATS: %s \n',name_feats)
                    
                    train_image_feats =  get_bags_of_sifts(...
                        train_image_paths,COLOUR_SPACE(1,c),DISTANCE,BIN_SIZE,FEATURE_STEP_SIZE,USE_PHOW,USE_NORM,vocab);
                    test_image_feats  = get_bags_of_sifts(...
                        test_image_paths,COLOUR_SPACE(1,c),DISTANCE,BIN_SIZE,FEATURE_STEP_SIZE,USE_PHOW,USE_NORM,vocab);
                     save(name_feats, 'train_image_feats','test_image_feats');
                 else
                    load(name_feats);
                end   

                for class=1 :size(CLASSIFIER,2)
                   if(CLASSIFIER(1,class) == "support vector machine" ) 
                       lbs  = [10.0,1.0,0.1,0.01,0.001,0.0001,0.00001...
                           0.00011,0.00001,0.00002,0.00003,0.00004,0.00005,];
                       for lam=1:size(lbs,2)
                        accuracy = SVM(train_image_feats, train_labels, test_image_feats,categories,lbs(1,lam),test_labels);
                        output = sprintf("%s, %d, %d, %d, %s, %d, %s, %f, %f \n",...
FEATURE,VOCAB_SIZE(1,v),FEATURE_STEP_SIZE,STEP_SIZE(1,ss),COLOUR_SPACE(1,c),USE_PHOW,CLASSIFIER(1,class),lbs(1,lam),accuracy);
                        
                        fprintf(fileID,output);
                       end
                   elseif(CLASSIFIER(1,class) == "nearest neighbor")
                       for nn=1:15
                        accuracy = kNN(nn,train_image_feats, train_labels, test_image_feats,categories,DISTANCE,test_labels);
                        output = sprintf("%s, %d, %d, %d, %s, %d, %s, %d, %f \n",...
FEATURE,VOCAB_SIZE(1,v),FEATURE_STEP_SIZE,STEP_SIZE(1,ss),COLOUR_SPACE(1,c),USE_PHOW,CLASSIFIER(1,class),nn,accuracy);
                        fprintf(fileID,output);
                       end
                   end
                end
            end
        end
       fprintf("done:%d/%d \n", v,size(VOCAB_SIZE,2));
    end    
end


function accuracy = kNN(k,train_image_feats, train_labels, test_image_feats,categories,DISTANCE,test_labels)
predicted_categories = nearest_neighbor_classify(k,train_image_feats, train_labels, test_image_feats,DISTANCE);

accuracy = get_acc(test_labels, categories,predicted_categories);
end


function accuracy = SVM(train_image_feats, train_labels, test_image_feats,categories,LAMBDA,test_labels)
predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats,LAMBDA);


accuracy = get_acc(test_labels, categories,predicted_categories);
end

function accuracy = get_acc( test_labels, categories,predicted_categories)
    num_categories = length(categories);    
    confusion_matrix = zeros(num_categories, num_categories);
    for i=1:length(predicted_categories)
        row = find(strcmp(test_labels{i}, categories));
        column = find(strcmp(predicted_categories{i}, categories));
        confusion_matrix(row, column) = confusion_matrix(row, column) + 1;
    end
    %if the number of training examples and test casees are not equal, this
    %statement will be invalid.
    num_test_per_cat = length(test_labels) / num_categories;
    confusion_matrix = confusion_matrix ./ num_test_per_cat;   
    accuracy = mean(diag(confusion_matrix));
end