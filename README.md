# Point Cloud Processing and Volume Calculation
This Python script allows you to load and process a point cloud, ultimately calculating and displaying the volume of its convex hull. 

To set up your Conda environment using the `environment.yml` file, follow these steps:

```bash
conda env create -f environment.yml
conda activate biomass
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

## Additional Configuration

For PyVista backend settings:

In your script, include:
``` python
import pyvista as pv
from pyvista import settings
settings.default_backend = 'ipyvtk'  # Set the default backend for PyVista
pv.set_jupyter_backend('trame')  # Set the Jupyter backend for PyVista
```
