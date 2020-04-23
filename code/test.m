function out = test(train_image_paths,test_image_paths,train_labels,categories,test_labels)
    out=1;
    FEATURE = ["bag of sift","spatial pyramids"];%'colour histogram', 'tiny image','bag of sift', 'spatial pyramids'
    
    COLOUR_SPACE =["GRAYSCALE","RGB"];%,HSV, GRAYSCALE, YCBR, NTSC
    DISTANCE = 'L1'; %L1,chisq need double quotes
    STEP_SIZE = 5;
    BIN_SIZE=6;
    FEATURE_STEP_SIZE = 5;
    VOCAB_SIZE = [10,20,30,50,100,200,400]; % you need to test the influence of this parameter
    MAX_LEVEL = 3;
    CLASSIFIER = ["support vector machine" ,"nearest neighbor"];%,'support vector machine';
    
    fileID = fopen('results.txt','a');
    for f=1:size(FEATURE,2)
        for v=1:size(VOCAB_SIZE,2)
            for c=1:size(COLOUR_SPACE,2)
                vocab = build_vocabulary(train_image_paths, VOCAB_SIZE(1,v),STEP_SIZE,COLOUR_SPACE(1,c),BIN_SIZE); %Also allow for different sift parameters

                if(FEATURE(1,f)=="bag of sift")
                    train_image_feats = get_bags_of_sifts(train_image_paths,COLOUR_SPACE(1,c),DISTANCE,BIN_SIZE,vocab); %Allow for different sift parameters
                    test_image_feats  = get_bags_of_sifts(test_image_paths, COLOUR_SPACE(1,c),DISTANCE,BIN_SIZE,vocab); 
                elseif(FEATURE(1,f)=="spatial pyramids")
                    train_image_feats = spatial_pyramid(train_image_paths,MAX_LEVEL,VOCAB_SIZE(1,v),DISTANCE,FEATURE_STEP_SIZE,COLOUR_SPACE(1,c),BIN_SIZE,vocab);
                    test_image_feats = spatial_pyramid(test_image_paths,MAX_LEVEL,VOCAB_SIZE(1,v),DISTANCE,FEATURE_STEP_SIZE,COLOUR_SPACE(1,c),BIN_SIZE,vocab);
                end
                
                for class=1 :size(CLASSIFIER,2)
                   if(CLASSIFIER(1,class) == "support vector machine" ) 
                       lbs  = [10.0,1.0,0.1,0.01,0.001,0.0001,0.00001];
                       for lam=1:size(lbs,2)
                        accuracy = SVM(train_image_feats, train_labels, test_image_feats,categories,lbs(1,lam),test_labels);
                        output = sprintf("%s, %d, %s, %s, %f, %f \n",FEATURE(1,f),VOCAB_SIZE(1,v),COLOUR_SPACE(1,c),CLASSIFIER(1,class),lbs(1,lam),accuracy);
                        fprintf(fileID,output);
                       end
                   elseif(CLASSIFIER(1,class) == "nearest neighbor")
                       for nn=1:15
                        accuracy = kNN(nn,train_image_feats, train_labels, test_image_feats,categories,DISTANCE,test_labels);
                        output = sprintf("%s, %d, %s, %s, %d, %f \n",FEATURE(1,f),VOCAB_SIZE(1,v),COLOUR_SPACE(1,c),CLASSIFIER(1,class),nn,accuracy);
                        fprintf(fileID,output);
                       end
                   end
        
                end
            end
        end
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