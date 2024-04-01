import argparse
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
from shapely.geometry import Polygon, Point
from lang_sam import LangSAM
import rerun as rr 


from collections import namedtuple

PointCloud = namedtuple("ColorGrid", ["positions", "colors"])


def PointCloudTuple(o3d_pcd):
    """Convert an Open3D PointCloud to a PointCloudTuple."""

    pcd_pts = np.asarray(o3d_pcd.points)
    pcd_rgb = np.asarray(o3d_pcd.colors) * 255

    return PointCloud(pcd_pts, pcd_rgb.astype(np.uint8))


def read_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors) * 255
    pcd_np = np.hstack((pcd_points, pcd_colors))
    return pcd_np, pcd

def cloud_to_image(pcd_np, image_width, image_height):
    min_coords = np.min(pcd_np[:, :2], axis=0)
    max_coords = np.max(pcd_np[:, :2], axis=0)
    scale_x = (image_width - 1) / (max_coords[0] - min_coords[0])
    scale_y = (image_height - 1) / (max_coords[1] - min_coords[1])
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    mapping = np.full((image_height, image_width), -1, dtype=int)
    for i, point in enumerate(pcd_np):
        x, y, _, r, g, b = point
        pixel_x = int((x - min_coords[0]) * scale_x)
        pixel_y = image_height - 1 - int((y - min_coords[1]) * scale_y)
        image[pixel_y, pixel_x] = [r, g, b]
        mapping[pixel_y, pixel_x] = i
    return image, mapping

def process_masks(orthoimage, text_prompt):
    model = LangSAM()
    orthoimage_pil = Image.fromarray(orthoimage)
    masks, _, _, _ = model.predict(orthoimage_pil, text_prompt)
    return [mask.squeeze().cpu().numpy() for mask in masks]

def crop_point_cloud(pcd_np, mapping, masks_np):
    polygons = []
    for mask in masks_np:
        mask_binary = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_points = contour.squeeze().tolist()
            if len(contour_points) > 2:
                polygons.append(Polygon(contour_points))
    h, w = mapping.shape[:2]
    points_inside = []
    for iy in range(h):
        for ix in range(w):
            point_index = mapping[iy, ix]
            if point_index != -1:
                point_2d = Point(ix, iy)
                if any(polygon.contains(point_2d) for polygon in polygons):
                    points_inside.append(point_index)
    cropped_point_cloud = pcd_np[points_inside]
    return cropped_point_cloud

def compute_volume(cropped_pcd_np):

    crop_pcd = o3d.geometry.PointCloud()

    crop_pcd.points = o3d.utility.Vector3dVector(cropped_pcd_np[:, :3])
    if cropped_pcd_np[:, 3:].max() > 1:
        pcd_crop_colors = cropped_pcd_np[:, 3:] / 255.0
    else:
        pcd_crop_colors = cropped_pcd_np[:, 3:]

    crop_pcd.colors = o3d.utility.Vector3dVector(pcd_crop_colors)

    hull = crop_pcd.compute_convex_hull()[0]
    hull.simplify_quadric_decimation(target_number_of_triangles=100)

    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))

    volume_m3 = hull.get_volume()

    conversion_factor = 35.3147
    volume_ft3 = volume_m3 * conversion_factor

    return volume_m3, volume_ft3, hull_ls, crop_pcd

def hull_lines(o3d_hull_lineset):

    vertices = np.asarray(o3d_hull_lineset.points)
    lines = np.asarray(o3d_hull_lineset.lines)
    hull_line_strips = []

    for i in range(len(lines)):
        line_strip = [vertices[lines[i, 0]].tolist(), vertices[lines[i, 1]].tolist()]
        hull_line_strips.append(line_strip)

    # Flatten the line strips list
    hull_line_strips = [point for strip in hull_line_strips for point in strip]

    return hull_line_strips

def log_results(orthoimage, volume_m3, volume_ft3, biomass_croped, wood_log_pcd, hull_2_line_strips):

    rr.init("biomass test", spawn=True)

    rr.log("wood stack", rr.Points3D(
    positions=wood_log_pcd.positions, 
    colors=wood_log_pcd.colors,
    labels=["original scan"]
    )
    )
    
    rr.log("cropped", rr.Points3D(
    positions=biomass_croped.positions, 
    colors=biomass_croped.colors,
    labels=["extracted biomass"]
    
    )
    )
    
    rr.log(
    "convex hull",
    rr.LineStrips3D(
        [hull_2_line_strips],
        colors=[[255, 0, 0]],  
        radii=[0.005],         # Line radius
        labels=["Convex Hull"]
    ),
    )
    rr.log("scan screenshot", rr.Image(data=orthoimage))
    rr.log("markdown", rr.TextDocument(
        f'## Biomass Volume:\n\n| Cubic Meters | Cubic Feet |\n| ------------ | ---------- |\n| {volume_m3:.2f}  | {volume_ft3:.2f} |',
        media_type=rr.MediaType.MARKDOWN))



def main(point_cloud_file, text_prompt):

    pcd_np, pcd = read_point_cloud(point_cloud_file)
    orthoimage, mapping = cloud_to_image(pcd_np, 512, 512)
    masks_np = process_masks(orthoimage, text_prompt)

    cropped_pcd_np = crop_point_cloud(pcd_np, mapping, masks_np)

    volume_m3, volume_ft3, o3d_hull_lineset, crop_pcd  = compute_volume(cropped_pcd_np)

    convex_hull = hull_lines(o3d_hull_lineset)

    biomass_pcd = PointCloudTuple(pcd)
    biomass_croped = PointCloudTuple(crop_pcd)

    log_results(orthoimage, volume_m3, volume_ft3, biomass_croped, biomass_pcd, convex_hull)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a point cloud file.')
    parser.add_argument('--file_path', type=str, help='Path to the point cloud file')
    parser.add_argument('--prompt', type=str, default='wood pile', help='Text prompt for mask prediction')
    args = parser.parse_args()

    main(args.file_path, args.prompt)
