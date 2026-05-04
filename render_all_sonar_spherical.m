filepath = dir("/media/seabrain/Expansion/output/*.hdf5");

filenames = strcat({filepath(:).folder}, "/", {filepath(:).name});

filenames = natsortfiles(filenames);

for i = 1:size(filepath,1) 
    disp(filenames{i})
    %[densityGrid, segmapGrid, ranges, azimuths] = 
    read_render_hdf5_spherical(filenames{i});
end