
% return concatenated color values
% append each color channel to a matrix
% images with 3 color channels, will be represented with a 2D
% matrix for feature extraction
function col_values  = get_color_values(I,COLORSPACE)
    
    switch(COLORSPACE)
        case "RGB"
            col_values = getColorMat(I);
        case "HSV"
            col_values = getColorMat(rgb2hsv(I));
        case "YCBR"
            col_values = getColorMat(rgb2ycbcr(I));
        case "NTSC"
            converted_img = rgb2ntsc(I);
            img = (converted_img + 1) / 2;
            col_values = getColorMat(rgb2ntsc(img));
    end
    col_values = single(col_values);
end

% concatenate the color values into a 2D Matrix
function col_values = getColorMat(img)
    c1 = img(:,:,1);
    c2 = img(:,:,2);
    c3 = img(:,:,3);
    
    col_values = [c1,c2,c3];
    
end


