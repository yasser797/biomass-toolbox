# Biomass-Toolbox

To set up your Conda environment using the `environment.yml` file, follow these steps:

```bash
conda env create -f environment.yml
conda activate biomass
```
## Usage

To run the script, use the following command:

```bash
python biomass_volume_rerun.py <path_to_point_cloud>
```

## How it Works

- The script loads a specified point cloud and performs voxel down-sampling.
- Then computes the axis-aligned bounding box of the point cloud and flips the point cloud.
- An alpha shape of the combined points (original and flipped) is computed.
- It generates random points on the faces of the alpha shape.
- The script calculates distances between the original point cloud and the alpha shape and masks the point cloud based on these distances.
- Outliers are removed and the convex hull is computed.
- The convex hull and the point cloud are visualized.
- Finally, the volume of the convex hull is calculated and displayed in both cubic meters and cubic feet.


