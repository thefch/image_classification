
% return PHOW descriptors for color images
% function that takes care of the SIFT descriptors extraction of colored
% images
function [frames, features] = get_phow(I,COLOR_SPACE,FEATURE_STEP_SIZE)
    switch(COLOR_SPACE)
            case "GRAYSCALE"
                color = "gray";
            case "RGB"
                color="rgb";
            case "HSV"
                color="hsv";
        otherwise
            print("Default ColorSpace(RGB) selected because %s not found \n",COLOR_SPACE);
            img = I;
    end
    img = single(I);
    
    % extract dense sift features in the specific colorspace
    [frames, features] = vl_phow(img,'Step',FEATURE_STEP_SIZE,'Color',color);
    features = single(features);
end
