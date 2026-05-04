function [densityGrid, segmapGrid, ranges, azimuths] = read_render_hdf5(filepath)
    %% File parts
    [~,fname,~] = fileparts(filepath);

    depth = h5read(filepath,"/distance").';
    intensity_orig = h5read(filepath,"/intensity").';
    segmap = h5read(filepath,"/instance_segmaps").';
    camera = jsondecode(h5read(filepath,"/camera"));
    maps = jsondecode(h5read(filepath,"/instance_attribute_maps"));

    imwrite(mat2gray(depth), strcat('output/', fname, '_depth_raw.png'), 'png'); %Azi and R: Saved as PNG properties
    imwrite(intensity_orig, strcat('output/', fname, '_intensity_raw.png'), 'png'); %Azi and R: Saved as PNG properties
    imwrite(segmap, jet(16), strcat('output/', fname, '_segmap_raw.png'), 'png'); %Azi and R: Saved as PNG properties
    %{
    %% Make colored segmap
    % recalculate
    nclasses = max([maps.idx])+1;
    colorss = turbo(nclasses);
    %% 
    
    %% Cartesian coordinates pointcloud plot
    [height, width] = size(depth);
    hfov = camera.horizontal_fov;
    vfov = camera.vertical_fov;
    azi = linspace(-hfov / 2, hfov / 2, width);
    ele = linspace(-vfov / 2, vfov / 2, height);
    [azi, ele] = meshgrid(azi, ele);
    r = depth;
   
    %% Time dependent gain
    intensity = intensity_orig.*(depth./50);

    %% Binning to create sonar image
    % Define bin edges for azimuth and range
    azimuth_bins = linspace(min(azi(:)), max(azi(:)), 513); % 512 bins for azimuth
    range_bins = linspace(0, 20, 513);                     % 512 bins for range with fixed min 0 and max 20
    intensity_bins = linspace(0, 10, 513);                   % 512 bins for intensity with limits from 0 to 1

    % Compute bin indices
    azimuth_idx = discretize(azi, azimuth_bins); % Retain original shape
    range_idx = discretize(r, range_bins);       % Retain original shape
    intensity_idx = discretize(intensity, intensity_bins); % Retain original shape
    
    % Remove NaN values from bin indices
    valid_idx = ~isnan(azimuth_idx) & ~isnan(range_idx) & ~isnan(intensity_idx);
    azimuth_idx = azimuth_idx(valid_idx);
    range_idx = range_idx(valid_idx);
    
    % Update data to only include valid indices
    azi = azi(valid_idx);
    r = r(valid_idx);
    intensity = intensity(valid_idx);
    segmap = segmap(valid_idx);
    
    % Define aggregation function with NaN cast
    agg = @(data, func) accumarray([range_idx(:), azimuth_idx(:)], data(:), [512, 512], func, cast(NaN, 'like', data));
    
    % Aggregate data using the defined function
    azimuthGrid = agg(azi, @(x) mean(x, 'omitnan'));
    rangeGrid = agg(r, @(x) mean(x, 'omitnan'));
    intensityGrid = agg(intensity, @sum);
    intensityGrid(isnan(intensityGrid)) = 0;
    segmapGrid = agg(segmap, @mode);
    
    %% Save output image
    azi_string = sprintf("%f:%f:%f", azimuth_bins(1), azimuth_bins(2)-azimuth_bins(1), azimuth_bins(end));
    r_string = sprintf("%f:%f:%f", range_bins(1), range_bins(2)-range_bins(1), range_bins(end));
    imwrite(intensityGrid, strcat('output/', fname, '_sonar.png'), 'png','Azimuth',azi_string,'Range',r_string); %Azi and R: Saved as PNG properties
    imwrite(segmapGrid, strcat('output/', fname, '_segmap.png'), 'png','Azimuth',azi_string,'Range',r_string); %Azi and R: Saved as PNG properties
   
    %% Export ranges
    ranges = movmean(range_bins, 2, 'Endpoints', 'discard');
    azimuths = movmean(azimuth_bins, 2, 'Endpoints', 'discard');

    %% Display
    fig(41);
    clf;
    densityGrid = intensityGrid;
    
    % Polar plot for intensity grid
    h1 = polarPcolor(ranges, rad2deg(azimuths), densityGrid, 'Nspokes',7,'autoOrigin','off','colbar',0); % Use the downloaded polarPcolor function
    
    set(gcf, 'Name', 'Sonar 2D with Intensity in Polar Coordinates');
    xlabel('Azimuth');
    ylabel('Range');
    cb1 = colorbar(gca,"westoutside");
    clim([0 1]);
    cb1.Label.String = 'Intensity (Normalized)';
    colormap('hot');

    set(gcf, 'Position', [100, 100, 700, 500]);
    exportgraphics(gcf, strcat('plot/', fname, '_sonarplot_polar.png'), 'Resolution', 300);
    
    % Plot segmap grid in polar coordinates
    fig(42);
    clf;
    h2 = polarPcolor(ranges, rad2deg(azimuths), segmapGrid,'Nspokes',7,'autoOrigin','off','ncolor',nclasses,'colormap','turbo','colbar',0); % Use the downloaded polarPcolor function
    set(gcf, 'Name', 'Sonar segmap in Polar Coordinates');
    xlabel('Azimuth');
    ylabel('Range');
    
    %colormap(colorss);
    cb2 = colorbar(gca,"westoutside");
    cb2.Label.String = 'Class';
    
    ticks = linspace(cb2.Limits(1), cb2.Limits(2), 1 + 2 * nclasses);
    ticks(1:2:end) = [];
    classlists = struct2cell(maps);
    labels(1) = {'NA'};
    labels([classlists{1,:}]+1) = classlists(2,:);
    cb2.Ticks = ticks;
    cb2.TickLabels = labels;
    
    % createlegend(colorss, )
    set(gcf, 'Position', [100, 600, 700, 500]);
    exportgraphics(gcf, strcat('plot/', fname, '_labelplot_polar.png'), 'Resolution', 300);
    drawnow();
    %}
end