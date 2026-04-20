import blenderproc as bproc
import bpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy_groupies as npg
from PIL import Image, PngImagePlugin
import matplotlib.pyplot as plt

context = bpy.context
wm = context.window_manager

if "new" in wm.keys():
    print("Not New")
    wm["new"] += 1
else:
    wm["new"] = 1
    print("New")
    bproc.loader.load_blend("raylength.blend",data_blocks=['objects','lights','worlds','scenes','cameras'],obj_types=['mesh','light','camera'])

def mode_omitnan(x):
    x = np.asarray(x)
    if x.size == 0:
        return np.nan
    if np.issubdtype(x.dtype, np.floating):
        x = x[~np.isnan(x)]
        if x.size == 0:
            return np.nan
    x = x.astype(np.int64, copy=False)
    mn = int(x.min())
    mx = int(x.max())
    c = np.bincount(x - mn, minlength=(mx - mn + 1))
    return (np.argmax(c) + mn)

objs = bproc.object.get_all_mesh_objects()

nframe = 2

#Reset frames
bpy.data.scenes["Scene"].frame_end = 0
bpy.data.scenes["Scene"].frame_start = 0
bproc.utility.reset_keyframes()

bpy.context.scene.render.use_compositing = True
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

meshes = bproc.filter.all_with_type(objs,bproc.types.MeshObject)

# Read the camera positions file and convert into homogeneous camera-world transformation
#with open(args.camera, "r") as f:
#    for line in f.readlines():
#        line = [float(x) for x in line.split()]
#        position, euler_rotation = line[:3], line[3:6]
#        matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
#        bproc.camera.add_camera_pose(matrix_world)

suzanne = bproc.filter.by_attr(meshes, "name", "Ground")
print(suzanne)
# Find point of interest, all cam poses should look towards it
poi = bproc.object.compute_poi(suzanne)
# Sample five camera poses
for i in range(nframe):
    # Sample random camera location above objects
    location = np.random.uniform([-10, -10, 2], [10, 10, 4])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.1, 0.1))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


#render
bproc.camera.set_resolution(1280, 720)
bpy.context.scene.render.resolution_percentage = 100 #make sure scene height and width are ok (edit)
bpy.context.scene.cycles.use_light_tree = False
bpy.context.scene.cycles.samples = 100

## NOW RENDER
bproc.renderer.set_output_format("OPEN_EXR", enable_transparency=False)  # Specify EXR format
data = bproc.renderer.render(output_key="tof")
#print(data['tof'][0].shape)
data2 = bproc.renderer.render_segmap(map_by=["instance", "cp_name_extra"],output_dir="segtest",default_values={"cp_name_extra": "NA"})
print(data2)

distance = [];
intensity = [];
camera = [];

# Sample five camera poses
for i in range(nframe):
    # Create a dictionary for camera data
    camera_dict = {
    'horizontal_fov': bpy.context.scene.camera.data.angle_x,
    'vertical_fov': bpy.context.scene.camera.data.angle_y
    }
    camera.append(camera_dict);
    ## add recast data
    distance.append(np.emath.logn(0.950, np.divide(data['tof'][i][:,:,1],data['tof'][i][:,:,2])))
    intensity.append(data['tof'][i][:,:,2])
   

data2['distance'] = distance;
data2['intensity'] = intensity;
data2['camera'] = camera;

print("SIZE OF SEGMAPS")
print(len(data2['instance_segmaps']))

#print(data2)

print("Creating sonar plots")
#Arrays for storage
sonar_bins_array = []
sonar_images = []
sonar_segmaps = []

for frame_idx in range(len(data2["distance"])):
    depth = data2["distance"][frame_idx]
    segmap = data2["instance_segmaps"][frame_idx]


    r = depth.astype(np.float64, copy=False)
    I0 = 1.0
    alpha = 4.5e-2 # Attenuation guess for 1.2 MHz sonar
    p = 2.0 # Spherical falloff
    r0 = 0.005
    intensity_orig = I0 * np.exp(-2.0 * alpha * r) / np.power(r + r0, p)

    cam = bpy.context.scene.camera
    hfov = cam.data.angle_x
    vfov = cam.data.angle_y

    # %% Cartesian coordinates pointcloud plot
    height, width = depth.shape
    azi = np.linspace(-hfov / 2.0, hfov / 2.0, width, dtype=np.float64)
    ele = np.linspace(-vfov / 2.0, vfov / 2.0, height, dtype=np.float64)
    azi, ele = np.meshgrid(azi, ele)
    r = depth

    # %% Time dependent gain
    intensity = intensity_orig * np.power((r + 0.005) / 5, 2.0)

    # %% Binning to create sonar image
    azimuth_bins = np.linspace(np.min(azi), np.max(azi), 513, dtype=np.float64)
    range_bins = np.linspace(0.0, 20.0, 513, dtype=np.float64)
    intensity_bins = np.linspace(0.0, 10.0, 513, dtype=np.float64)

    azimuth_idx = np.digitize(azi, azimuth_bins) - 1
    range_idx = np.digitize(r, range_bins) - 1
    intensity_idx = np.digitize(intensity, intensity_bins) - 1

    valid_idx = (
        (azimuth_idx >= 0) & (azimuth_idx < 512) &
        (range_idx >= 0) & (range_idx < 512) &
        (intensity_idx >= 0) & (intensity_idx < 512)
    )

    azimuth_idx = azimuth_idx[valid_idx].astype(np.int64, copy=False)
    range_idx = range_idx[valid_idx].astype(np.int64, copy=False)

    azi = azi[valid_idx]
    r = r[valid_idx]
    intensity = intensity[valid_idx]
    segmap = segmap[valid_idx]


    # %% numpy groupies aggregation (accumarray equivalent)
    lin_size = 512 * 512
    lin = range_idx * 512 + azimuth_idx

    azimuthGrid = npg.aggregate(lin, azi, size=lin_size, func="nanmean", fill_value=np.nan).reshape(512, 512)
    rangeGrid = npg.aggregate(lin, r, size=lin_size, func="nanmean", fill_value=np.nan).reshape(512, 512)
    intensityGrid = npg.aggregate(lin, intensity, size=lin_size, func="nansum", fill_value=0).reshape(512, 512)
    segmapGrid = npg.aggregate(lin, segmap, size=lin_size, func=mode_omitnan, fill_value=np.nan).reshape(512, 512)

    # %% Save output image
    #azi_string = f"{azimuth_bins[0]}:{azimuth_bins[1] - azimuth_bins[0]}:{azimuth_bins[-1]}"
    #r_string = f"{range_bins[0]}:{range_bins[1] - range_bins[0]}:{range_bins[-1]}"

    #meta = PngImagePlugin.PngInfo()
    #meta.add_text("Azimuth", azi_string)
    #meta.add_text("Range", r_string)

    print(intensityGrid)

    #int_img_u8 = (np.clip(intensityGrid.astype(np.float64, copy=False), 0.0, 1.0) * 255.0).astype(np.uint8)
    #seg_img_u8 = segmapGrid.astype(np.uint8, copy=False)

    #Image.fromarray(int_img_u8, mode="L").save(f"output/{fname}_sonar.png", pnginfo=meta)
    #Image.fromarray(seg_img_u8, mode="L").save(f"output/{fname}_segmap.png", pnginfo=meta)


    # %% Export ranges
    azimuth_centers = 0.5 * (azimuth_bins[:-1] + azimuth_bins[1:])
    range_centers = 0.5 * (range_bins[:-1] + range_bins[1:])

    sonar_bins = {
        "azimuth_bins": azimuth_bins.tolist(),
        "range_bins": range_bins.tolist(),
        "azimuth_centers": azimuth_centers.tolist(),
        "range_centers": range_centers.tolist()
    }

    # --- Save sonar images ---
    azi_string = f"{azimuth_bins[0]}:{azimuth_bins[1] - azimuth_bins[0]}:{azimuth_bins[-1]}"
    r_string = f"{range_bins[0]}:{range_bins[1] - range_bins[0]}:{range_bins[-1]}"

    meta = PngImagePlugin.PngInfo()
    meta.add_text("Azimuth", azi_string)
    meta.add_text("Range", r_string)

    int_img_u8 = (np.clip(intensityGrid.astype(np.float64, copy=False), 0.0, 1.0) * 255.0).astype(np.uint8)
    seg_img_u16 = segmapGrid.astype(np.uint16, copy=False)

    fname = f"test_sonar_{frame_idx:06d}"

    Image.fromarray(int_img_u8, mode="L").save(f"output/{fname}_sonar.png", pnginfo=meta)
    Image.fromarray(seg_img_u16, mode="I;16").save(f"output/{fname}_segmap.png", pnginfo=meta)

    # --- Per frame bin JSON entry ---
    sonar_bins_array.append({
        "frame_idx": frame_idx,
        "azimuth_bins": azimuth_bins.tolist(),
        "range_bins": range_bins.tolist(),
        "azimuth_centers": azimuth_centers.tolist(),
        "range_centers": range_centers.tolist()
    })
    
    sonar_images.append(intensityGrid.astype(np.float32, copy=False))
    sonar_segmaps.append(segmapGrid.astype(segmap.dtype, copy=False))

sonar_bins = {"sonar_bins": sonar_bins_array}
sonar_images = {"sonar_intensity": sonar_images}
sonar_segmaps = {"sonar_segmaps": sonar_segmaps}

# %% Display
#densityGrid = intensityGrid
#AZ, RR = np.meshgrid(np.deg2rad(azimuths), ranges)

#fig = plt.figure(41)
#fig.clf()
#ax = fig.add_subplot(111, projection="polar")
#pc = ax.pcolormesh(AZ, RR, densityGrid, shading="auto", cmap="hot")
#ax.set_title("Sonar 2D with Intensity in Polar Coordinates")
#ax.set_xlabel("Azimuth")
#ax.set_ylabel("Range")
#cb1 = fig.colorbar(pc, ax=ax, location="left")
#pc.set_clim(0.0, 1.0)
#cb1.set_label("Intensity (Normalized)")
#plt.show()


bproc.writer.write_hdf5("output",{**data, **data2, **sonar_images, **sonar_segmaps, **sonar_bins})


print("Finished rendering");

# Render and save as PNG
#print(distance)
#points, intensity_scaled = create_point_cloud(intensity, distance)
#print("points")
#print(points)
#print("intensity")
#print(intensity_scaled)
#density_grid = downsample_point_cloud_binning(points, intensity_scaled)
#print("density_grid")
#print(density_grid)
#plot_density_grid(density_grid)

#Write output
#bproc.writer.write_png_segm(, data["segmentation"])
#bproc.writer.write_openexr_rgba(.exr", data["colors"])

#get the pixels and put them into a numpy array
#pixels = np.array(bpy.data.images['Viewer Node'].pixels)
#print(len(pixels))

#width = bpy.context.scene.render.resolution_x 
#height = bpy.context.scene.render.resolution_y

#reshaping into image array 4 channel (rgbz)
#image = pixels.reshape(height,width,4)

#depth analysis...
#distance = image[:,:,3]
#zf = z[z<1000] #
#print(np.min(zf),np.max(zf))

#intensity analysis
#intensity = image[:,:,1]
#print(np.min(i),np.max(i))
