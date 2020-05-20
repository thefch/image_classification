function output = nearest_neighbor_classify(k,train_image_feats, train_labels, test_image_feats,DISTANCE)
    [test_rows test_cols] = size(test_image_feats);
    %[train_rows train_cols] = size(test_image_feats);
    
    
	labels = init_labels(train_labels);  
    predictions = strings(test_rows,1);
    for i=1: test_rows
        test_image = test_image_feats(i,:);
        if (DISTANCE == "chisq" || DISTANCE == "L1" || DISTANCE == "emd")
            DISTANCE = char(DISTANCE);
            diff_list = custom_pdist2(test_image,train_image_feats,DISTANCE);    
        else
            diff_list = pdist2(test_image,train_image_feats,DISTANCE);    
        end
		
        
        diff_list = diff_list(:);
        table = [diff_list,labels];
	
        table = sortrows(table,1);
    
		table = table(1:k,:);
    
  
        prediction_label_id = mode(table(:,2));
		predicted_label = train_labels{prediction_label_id*100};
        predictions(i,:) = predicted_label;
%         if (mod(i, 200) == 0)
%             fprintf("Completed training example: %d \n", i);
%         end

    end
    
    
    output=predictions;
end

function output = init_labels(train_labels)
    %labels = train_labels.';
    labels = unique(train_labels);
    labels = zeros(length(labels),1);
    prev = 1;
    for i =1: length(labels)
        labels(prev:i*100,1) = i;
        prev = i*100;
    end
    output = labels;
end

