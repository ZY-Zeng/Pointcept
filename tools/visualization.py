import open3d as o3d
from pathlib import Path
import torch
import numpy as np
import os

from matplotlib import pyplot as plt
from scipy import stats

_label_to_color_uint8 = {
    0: [255, 0, 0],  # rebar
    1: [0, 255, 0],  # sleeve
    2: [0, 0, 255],  # other
}

_label_to_color = dict([
    (label, (np.array(color_uint8).astype(np.float64) / 255.0).tolist())
    for label, color_uint8 in _label_to_color_uint8.items()
])

_name_to_color_uint8 = {
    "rebar": [255, 0, 0],  # rebar
    "sleeve": [0, 255, 0],  # sleeve
    "other": [0, 0, 255],  # other
}

_name_to_color = dict([(name, np.array(color_uint8).astype(np.float64) / 255.0)
                       for name, color_uint8 in _name_to_color_uint8.items()])


def load_real_data(npy_path):
    """
    Args:
        pth_path: Path to the .pth file.
    Returns:
        points: (N, 3), float64
        colors: (N, 3), float64, 0-1
        labels: (N, ), int64, {1, 2, ..., 36, 39, 255}.
    """
    # - points: (N, 3), float32           -> (N, 3), float64
    # - colors: (N, 3), float32, 0-255    -> (N, 3), float64, 0-1
    # - labels: (N, 1), float64, 0-19,255 -> (N,  ), int64, 0-19,255

    points = np.load(Path(npy_path) / "coord.npy")
    colors = np.load(Path(npy_path) / "color.npy")
    labels = np.load(Path(npy_path) / "segment.npy")
    points = points.astype(np.float64)
    colors = colors.astype(np.float64) / 255.0
    assert len(points) == len(colors) == len(labels)

    labels = labels.astype(np.int64).squeeze()
    return points, colors, labels


def load_pred_labels(label_path):
    """
    Args:
        label_path: Path to the .txt file.
    Returns:
        labels: (N, ), int64, {1, 2, ..., 36, 39}.
    """
    def read_labels(label_path):
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                labels.append(int(line.strip()))
        return np.array(labels)

    return np.array(read_labels(label_path))


def render_to_image(pcd, save_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)


def visualize_scene_by_path(data_path, label_path, save_as_image=False):
    label_dir = Path("exp/pcc/result")

    print(f"Visualizing {data_path}")

    # Load pcd and real labels.

    points, colors, real_labels = load_real_data(data_path)

    # Visualize rgb colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if save_as_image:
        render_to_image(pcd, f"image/{data_path.stem}_rgb.png")
    else:
        o3d.visualization.draw_geometries([pcd], window_name="RGB colors")

    # Visualize real labels
    real_label_colors = np.array([_label_to_color[l] for l in real_labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(real_label_colors)
    if save_as_image:
        render_to_image(pcd, f"image/{data_path.stem}_real.png")
    else:
        o3d.visualization.draw_geometries([pcd], window_name="Real labels")

    # Load predicted labels
    pred_labels = np.load(Path(label_dir) / f"{label_path}")
    pred_label_colors = np.array([_label_to_color[l] for l in pred_labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pred_label_colors)
    stem, _ = os.path.splitext(label_path)
    o3d.io.write_point_cloud(f"exp/pcc/result/ply/{stem}.ply", pcd)

    if save_as_image:
        render_to_image(pcd, f"image/{scene_path.stem}_pred.png")
    else:
        o3d.visualization.draw_geometries([pcd], window_name="Pred labels")

def visualize_scene_by_name(npy_path, save_as_image=False):
    area_name, scene_name = scene_path(npy_path)
    data_root = Path("data") / "pcc" / f"{area_name}" / f"{scene_name}"
    visualize_scene_by_path(data_root, npy_path, save_as_image=save_as_image)

    
def scene_path(npy_path):
    fname = Path(npy_path).stem 
    base = fname.removesuffix("_pred") 
    area, scene = base.split("-", 1)   
    return area, scene  

if __name__ == "__main__":
    # Used in main text
    # hallway_10
    # lobby_1
    # office_27
    # office_30

    # Use in supplementary
    # visualize_scene_by_name("conferenceRoom_2")
    # visualize_scene_by_name("office_35")
    # visualize_scene_by_name("office_18")
    # visualize_scene_by_name("office_5")
    # visualize_scene_by_name("office_28")
    # visualize_scene_by_name("office_3")
    # visualize_scene_by_name("hallway_12")
    visualize_scene_by_name("Floor_4-pcz_9_pred.npy")
    visualize_scene_by_name("Floor_4-pcz_3_pred.npy")
    visualize_scene_by_name("Floor_4-pcz_5_pred.npy")
    visualize_scene_by_name("Floor_4-pcz_6_pred.npy")

    # Visualize all scenes
    # data_root = Path("data") / "scannetv2" / "val"
    # scene_paths = sorted(list(data_root.glob("*.pth")))
    # scene_names = [p.stem for p in scene_paths]
    # for scene_name in scene_names:
    #     visualize_scene_by_name(scene_name, save_as_image=True)