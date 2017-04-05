
function out = hist_match(REF,QUERY)

    M = zeros(256,1,'uint8'); %// Store mapping - Cast to uint8 to respect data type
    hist1 = imhist(QUERY);
    hist2 = imhist(REF);
    cdf1 = cumsum(hist1) / numel(QUERY);
    cdf2 = cumsum(hist2) / numel(REF);

    %// Compute the mapping
    for idx = 1 : 256
        [~,ind] = min(abs(cdf1(idx) - cdf2));
        M(idx) = ind-1;
    end

    %// Now apply the mapping to get first image to make
    %// the image look like the distribution of the second image
    out = M(double(QUERY)+1);

end
