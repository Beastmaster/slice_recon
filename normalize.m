




function img = normalize(img)
img = double(img);
mmax = max(img(:));
mmin = min(img(:));
mmean = mean(img(:));

img = (img-mmin)*256/(mmax-mmin);

%img = (img-mmean)/sstd;
end

