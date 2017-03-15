function maskedVolume = addMask(volume, mask,p)
cvolume=repmat(volume,[1,1,1,3]);
m = max(volume(:));

mask = logical(mask);

r=cat(4,ones(size(volume)),zeros(size(volume)),zeros(size(volume)));
g=cat(4,zeros(size(volume)),ones(size(volume)),zeros(size(volume)));
b=cat(4,zeros(size(volume)),zeros(size(volume)),ones(size(volume)));

cvolume(repmat(mask,[1,1,1,3])) = cvolume(repmat(mask,[1,1,1,3]))+...
  r(repmat(mask,[1,1,1,3]))*m*p(1);
cvolume(repmat(mask,[1,1,1,3])) = cvolume(repmat(mask,[1,1,1,3]))+...
  g(repmat(mask,[1,1,1,3]))*m*p(2);
cvolume(repmat(mask,[1,1,1,3])) = cvolume(repmat(mask,[1,1,1,3]))+...
  b(repmat(mask,[1,1,1,3]))*m*p(3);

maskedVolume = cvolume;

end

