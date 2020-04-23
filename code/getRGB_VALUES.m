function img = getRGB_VALUES(I)
 
%      I = imread(image_paths{1});
%     imshow(I);
%     
     img = im2single(I);
    img = getValues(img);
    % DESCRIPTORS from sift features of the specific image
%      [~, feats] = vl_dsift(img,'Fast','Step',STEP_SIZE) ;
    
end

function new_img = getValues(img)
    [sizey,sizex,~] = size(img);
    new_img = zeros(sizey,sizex);
    for y=1:sizey
        for x=1:sizex
            red = img(y,x,1);
            green = img(y,x,2);
            blue = img(y,x,3);
%             fprintf("%d %d %d \n",red,green,blue);
            
            m = (red+green+blue);
            new_img(y,x) = m;
            
        end
    end
    
    new_img = new_img ./ max(new_img(:));
%     disp(new_img);
    new_img = single(new_img);

end