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
            % because some values are 0, we need to avoid that
            img = (converted_img + 1) / 2;
            col_values = getColorMat(rgb2ntsc(img));
    end
    col_values = single(col_values);
end

function col_values = getColorMat(img)
    red = img(:,:,1);
    green = img(:,:,2);
    blue = img(:,:,3);
    
    col_values = [red,green,blue];
    
end


