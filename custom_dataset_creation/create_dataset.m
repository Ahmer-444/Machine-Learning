function create_dataset(images_dir,sz)
    count = 0;
    images_list = dir(images_dir);
    data_array = zeros(sz(1),sz(2),3,size(images_list,1)-2);
    size(data_array)
    for i = 3:size(images_list,1)
        I = imread(strcat(images_dir,images_list(i).name));
        resized = imresize(I,sz);
        normalized = im2double(resized);
        data_array(:,:,:,i-2) = normalized;
    end
    save 'SideFootData.mat' data_array;
end