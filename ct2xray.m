% Author: QIN Shuo
% Date:   2017/5/10
%
% 
%

%% Simulate X ray image from CT image

ct_im = 'data/3.nii';
ct = load_nii(ct_im); ct = ct.img;

% sum along axis
th=800;
ct = ct -th;
ct(ct<0)=0;

i=500;

for i=200:5:500
slice = ct(:,:,i);
slice = normalize(slice);
slice = uint16(imresize(slice,[400 400]));

x_sum = sum(slice,1);
x_rep = uint16(repmat(x_sum,length(x_sum),1));
y_sum = sum(slice,2);
y_rep = uint16(repmat(y_sum,1,length(y_sum)));

imwrite(x_rep,fullfile('data/train',strcat('x_',num2str(i),'.png')));
imwrite(y_rep,fullfile('data/train',strcat('y_',num2str(i),'.png')));
imwrite(slice,fullfile('data/train',strcat('im',num2str(i),'.png')));

end

return;


subplot(1,3,1);
imshow(x_rep,[]);
subplot(1,3,2);
imshow(y_rep,[]);
subplot(1,3,3);
imshow(slice,[]);








