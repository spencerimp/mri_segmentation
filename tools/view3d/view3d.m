function varargout = view3d(varargin)
% Display 3D Volume slice by slice
%
%      VIEW3D creates a new VIEW3D with a random dummy volume.
%
%      H = VIEW3D returns the handle to a new VIEW3D
%
%      VIEW3D(vol) creates a new VIEW3D and displays the 3D volume vol.
%      VIEW3D(vol,'dontNormalize') no normalization
%      VIEW3D(vol,title) Adds a title to the window
%      VIEW3d(vol,'defaultOrientation', x) Sets the default orientation to x
%
%      See also view3dCloseAll.
%
%                 2003-2009, Erik Dam, Nordic Bioscience

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @view3d_OpeningFcn, ...
                   'gui_OutputFcn',  @view3d_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before view3d is made visible.
function view3d_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to view3d (see VARARGIN)

% Choose default command line output for view3d
handles.output = hObject;

if evalin('base','exist(''view3d_handles'')')
    handleList = evalin('base','view3d_handles');
else
    handleList = [];
end
handleList = [handleList, hObject];
assignin('base', 'view3d_handles', handleList)

if isempty(varargin)
  disp('Creating dummy image')
  im = rand(20,30,4);
else
  im = varargin{1};
  if ndims(im)~=3	
		if ndims(im)==2
			sz = size(im);
			sz(3) = 1;
			im = reshape(im,sz);
			disp('Input volume is supposed to be 3D')
    elseif ndims(im)==4
      disp('Attempting to display the volume in color')
    else
			error('Input volume is supposed to be 3D')
		end
	end
	if min(size(im)) <= 1
		warning('Input volume has side shorter than 2')
	end
end
normalize = 1;
rotateSlices = false;
defaultOrientation = 2;
skipArg = false;
for i = 2:length(varargin)
    if skipArg,
        skipArg = false;
        continue;
    end
    
  str = varargin{i};
  if ~ischar(str)
    disp(['Ignoring input parameter ',num2str(i),':'])
    varargin{i};
  end
  switch str
   case 'defaultOrientation'
       defaultOrientation = varargin{i+1};
       skipArg = true;
    case 'rotateSlices'
      rotateSlices = true;
   case 'dontNormalize'
    normalize = 0;
   otherwise
    set(handles.title, 'String', str);
    set(hObject,'Name',['View3D - ',str])
  end
end
handles.normalize = normalize;
if normalize
   handles.imMax = max(im(:));
   handles.imMin = min(im(:));
end

%rotation
if rotateSlices,
  switch defaultOrientation
    case 1
      im = permute(im,[1,3,2]);
    case 2
      im = permute(im,[3,2,1]);
    case 3
      im = permute(im,[2,1,3]);
  end
end

% Orientation
handles.orientation = defaultOrientation;
set(handles.tagZ,'selected','on')
% Slice
handles.slices = size(im,handles.orientation);
handles.slice  = 0;
handles.volume = im;
if handles.slices==1
	set(handles.sliceSlider,'Visible','off')
else
	set(handles.sliceSlider, 'Min',1, 'Max',handles.slices, 'Value',1, ...
		'SliderStep',[1/handles.slices 10/handles.slices])
end
guidata(hObject, handles); % Update handles structure
updateSlice(handles, round(handles.slices / 2));

% %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % % 
% The display function
% %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % % 
 
function done = updateSlice(handles, slice)
  done = 0;
  if slice < 1 || slice > handles.slices
    disp(['The slice number must be an integer between 1 and ',num2str(handles.slices)])
    return
  end
  if slice == handles.slice
    disp('no update')
    return
  end
  set(handles.sliceSlider, 'Value' , slice)
  set(handles.sliceNo    , 'String', num2str(slice))
  im = GetIm(handles);
  axes(handles.image);
  imshow(im)
  % guidata(handles.image, handles); % Update handles structure
  done = 1;

function im = GetIm(handles)
  slice = get(handles.sliceSlider, 'Value');
  if ndims(handles.volume) == 3
    switch handles.orientation
      case 1, im = squeeze(handles.volume(slice,:,:));
      case 2, im = squeeze(handles.volume(:,slice,:));
      case 3, im = squeeze(handles.volume(:,:,slice));
    end
  elseif ndims(handles.volume) == 4
    switch handles.orientation
      case 1, im = squeeze(handles.volume(slice,:,:,:));
      case 2, im = squeeze(handles.volume(:,slice,:,:));
      case 3, im = squeeze(handles.volume(:,:,slice,:));
    end
  else error('Wrong dimensionality of volume');
  end
  if handles.normalize
    ma = handles.imMax;
    mi = handles.imMin;
    if ~isfloat(im)
      im = single(im);
      mi = single(mi);
      ma = single(ma);
    end
    if (ma>mi)
      im = (im-mi)/(ma-mi);
    end
  end
 
% %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % % 
% Callback functions
% %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % %  % % 

function sliceNo_Callback(hObject, eventdata, handles)
	str = get(hObject,'String');
	slice = str2double(str);
	if ~isnumeric(slice)
		disp('The slice must be a input as a number')
		return
	end
	if round(slice) ~= slice
		disp('The slice number must be an integer')
		return
	end
	updateSlice(handles, slice);

function sliceSlider_Callback(hObject, eventdata, handles)
	updateSlice(handles, round(get(hObject, 'Value')));

function varargout = view3d_OutputFcn(hObject, eventdata, handles)
	varargout{1} = handles.output;

function sliceNo_CreateFcn(hObject, eventdata, handles)
	if ispc
		set(hObject,'BackgroundColor','white');
	else
		set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
	end

function sliceSlider_CreateFcn(hObject, eventdata, handles)
	usewhitebg = 1;
	if usewhitebg
		set(hObject,'BackgroundColor',[.9 .9 .9]);
	else
		set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
	end

function buttonX(hObject, eventdata, handles)
	setOrientation(hObject, handles, 1)
function buttonY(hObject, eventdata, handles)
	setOrientation(hObject, handles, 2)
function buttonZ(hObject, eventdata, handles)
	setOrientation(hObject, handles, 3)

function setOrientation(hObject, handles, orient)
	tags = [handles.tagX,handles.tagY,handles.tagZ];
	set(tags(orient),'selected','on');
	set(tags(setdiff(1:3,orient)),'selected','off');
	handles.orientation = orient;
	% slice slider ui
	handles.slice = 0;
	handles.slices = size(handles.volume, orient);
	set(handles.sliceSlider, 'Min',1, 'Max',handles.slices, 'Value',1, ...
		  'SliderStep',[1/handles.slices 10/handles.slices])
	guidata(hObject, handles);
	updateSlice(handles, round(handles.slices / 2));

	
function button3D(hObject, eventdata, handles)
  Visualize3D(handles, 0.5)

function button3Dplus(hObject, eventdata, handles)
  Visualize3D(handles, 1.5)

function Visualize3D(handles, smoothing)
  figure
  vol = handles.volume;
  sz = size(vol);
  x1=sz(1); x2=0;
  y1=sz(2); y2=0;
  z1=sz(3); z2=0;
  % crop to around binary
  for x = 1:sz(1), for y = 1:sz(2), for z = 1:sz(3)
           if vol(x,y,z)>0
              if x<x1, x1=x; end, if x>x2, x2=x; end;
              if y<y1, y1=y; end, if y>y2, y2=y; end;
              if z<z1, z1=z; end, if z>z2, z2=z; end;
           end
        end, end, end
  if x1>x2 || y1>y2 || z1>z2
     disp('Found no data to visualize')
  end
  x1 = max(1,x1-1); x2 = min(sz(1),x2+1);
  y1 = max(1,y1-1); y2 = min(sz(2),y2+1);
  z1 = max(1,z1-1); z2 = min(sz(3),z2+1);
  vol = vol(x1:x2, y1:y2, z1:z2);
  % visualize
  structs = unique(vol);
  for s = 1:length(structs)
     struct = structs(s);
     if ~struct, continue, end
     segmentation = smooth3(vol==struct, 'gaussian', 3, smoothing);
     p = patch(isosurface(segmentation,0.5));
     isonormals(segmentation,p);
     col = [0.8,.75,.65]+0.2*(rand(1,3)-0.5);
     col = col(randperm(3));
     set(p,'FaceColor',col,'EdgeColor','none','AmbientStrength',.7);
     set(p,'SpecularExponent',20, 'SpecularStrength',0.7);
  end
  % daspect(mes.^-1)
  view(3);
  axis tight
  axis off
  l = light;
  light('Position', -get(l,'Position'))
  lighting phong % gouraud
  rotate3d('on')
  set(gcf,'Renderer','OpenGL')


function buttonSave(hObject, eventdata, handles)
  im = GetIm(handles);
  imwrite(im,'view3d.tif')
  
