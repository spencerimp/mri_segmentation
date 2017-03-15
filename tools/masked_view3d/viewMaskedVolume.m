function [pp]=viewMaskedVolume(volume, mask, tit)
%kk=addMask(volume, mask);

LOT = importdata('LOT.mat');
kk = zeros([size(volume) 3]);

for i=1:134
    idx = find(mask==i);
    [x,y,z] = ind2sub(size(mask),idx);
    
    for j = 1:length(x)
        kk(x(j),y(j),z(j),:) = [LOT(i,:)];
    end
end

v1 = mask==0;
v1 = repmat(v1.*volume, [1 1 1 3]);
v1 = v1./max(v1(:));
pp = kk+v1;

view3d(kk+v1, tit, 'dontNormalize')
end


