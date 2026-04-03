import torch
import flash_gaussian_splatting

import csv
import os
import sys
import json
import time


class Scene:
    def __init__(self, device):
        self.device = device
        self.num_vertex = 0
        self.position = None
        self.shs = None
        self.opacity = None
        self.cov3d = None

    def loadPly(self, scene_path):
        self.num_vertex, self.position, self.shs, self.opacity, self.cov3d = (
            flash_gaussian_splatting.ops.loadPly(scene_path)
        )
        # 58*4byte
        self.position = self.position.to(self.device)  # 3
        self.shs = self.shs.to(self.device)  # 48
        self.opacity = self.opacity.to(self.device)  # 1
        self.cov3d = self.cov3d.to(self.device)  # 6


class Camera:
    def __init__(self, camera_json, render_scale=1.0):
        self.id = camera_json["id"]
        self.img_name = camera_json["img_name"]
        self.width = max(1, int(round(camera_json["width"] * render_scale)))
        self.height = max(1, int(round(camera_json["height"] * render_scale)))
        self.position = torch.tensor(camera_json["position"])
        self.rotation = torch.tensor(camera_json["rotation"])
        self.focal_x = camera_json["fx"] * render_scale
        self.focal_y = camera_json["fy"] * render_scale
        self.zFar = 100.0
        self.zNear = 0.01


# 静态分配内存光栅化器
class Rasterizer:
    # 构造函数中分配内存
    def __init__(self, scene, MAX_NUM_RENDERED, MAX_NUM_TILES):
        # 24 bytes
        self.gaussian_keys_unsorted = torch.zeros(
            MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64
        )
        self.gaussian_values_unsorted = torch.zeros(
            MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32
        )
        self.gaussian_keys_sorted = torch.zeros(
            MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64
        )
        self.gaussian_values_sorted = torch.zeros(
            MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32
        )

        self.MAX_NUM_RENDERED = MAX_NUM_RENDERED
        self.MAX_NUM_TILES = MAX_NUM_TILES
        self.SORT_BUFFER_SIZE = flash_gaussian_splatting.ops.get_sort_buffer_size(
            MAX_NUM_RENDERED
        )
        self.list_sorting_space = torch.zeros(
            self.SORT_BUFFER_SIZE, device=scene.device, dtype=torch.int8
        )
        self.ranges = torch.zeros(
            (MAX_NUM_TILES, 2), device=scene.device, dtype=torch.int32
        )
        self.curr_offset = torch.zeros(1, device=scene.device, dtype=torch.int32)

        # 40 bytes
        self.points_xy = torch.zeros(
            (scene.num_vertex, 2), device=scene.device, dtype=torch.float32
        )
        self.rgb_depth = torch.zeros(
            (scene.num_vertex, 4), device=scene.device, dtype=torch.float32
        )
        self.conic_opacity = torch.zeros(
            (scene.num_vertex, 4), device=scene.device, dtype=torch.float32
        )

    # 前向传播（应用层封装）
    def forward(self, scene, camera, bg_color):
        # 属性预处理 + 键值绑定
        self.curr_offset.fill_(0)
        flash_gaussian_splatting.ops.preprocess(
            scene.position,
            scene.shs,
            scene.opacity,
            scene.cov3d,
            camera.width,
            camera.height,
            16,
            16,
            camera.position,
            camera.rotation,
            camera.focal_x,
            camera.focal_y,
            camera.zFar,
            camera.zNear,
            self.points_xy,
            self.rgb_depth,
            self.conic_opacity,
            self.gaussian_keys_unsorted,
            self.gaussian_values_unsorted,
            self.curr_offset,
        )

        # 键值对数量判断 + 处理键值对过多的异常情况
        num_rendered = int(self.curr_offset.cpu()[0])
        # print(num_rendered)
        if num_rendered >= self.MAX_NUM_RENDERED:
            raise "Too many k-v pairs!"

        flash_gaussian_splatting.ops.sort_gaussian(
            num_rendered,
            camera.width,
            camera.height,
            16,
            16,
            self.list_sorting_space,
            self.gaussian_keys_unsorted,
            self.gaussian_values_unsorted,
            self.gaussian_keys_sorted,
            self.gaussian_values_sorted,
        )
        # 排序 + 像素着色 + 混色阶段
        out_color = torch.zeros(
            (camera.height, camera.width, 3), device=scene.device, dtype=torch.int8
        )
        flash_gaussian_splatting.ops.render_16x16(
            num_rendered,
            camera.width,
            camera.height,
            self.points_xy,
            self.rgb_depth,
            self.conic_opacity,
            self.gaussian_keys_sorted,
            self.gaussian_values_sorted,
            self.ranges,
            bg_color,
            out_color,
        )
        return out_color


def savePpm(image, path):
    image = image.cpu()
    assert image.dim() >= 3
    assert image.size(2) == 3
    with open(path, "wb") as f:
        f.write(
            b"P6\n"
            + f"{image.size(1)} {image.size(0)}\n255\n".encode()
            + image.numpy().tobytes()
        )


def benchmark_model(model_path):
    print(f"Benchmarking {model_path}")
    scene_path = os.path.join(
        model_path, "point_cloud", "iteration_30000", "point_cloud.ply"
    )
    camera_path = os.path.join(model_path, "cameras.json")
    device = torch.device("cuda:0")
    bg_color = torch.zeros(3, dtype=torch.float32)  # black
    render_scale = 0.25

    scene = Scene(device)
    scene.loadPly(scene_path)

    with open(camera_path, "r") as camera_file:
        cameras_json = json.loads(camera_file.read())
    cameras = [
        Camera(camera_json, render_scale=render_scale)
        for i, camera_json in enumerate(cameras_json)
        if i % 8 == 0
    ]

    MAX_NUM_RENDERED = 2**27
    MAX_NUM_TILES = 2**20
    rasterizer = Rasterizer(scene, MAX_NUM_RENDERED, MAX_NUM_TILES)
    fps_values = []

    for warmup in range(2):
        if warmup == 0:
            print("Warmup...")
        else:
            print("Actual...")
        for _ in range(200):
            for i, camera in enumerate(cameras):
                if warmup == 0:
                    image = rasterizer.forward(scene, camera, bg_color)  # warm up
                else:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    image = rasterizer.forward(scene, camera, bg_color)
                    torch.cuda.synchronize()
                    t1 = time.time()

                    fps_values.append(1 / (t1 - t0))

    average_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
    return average_fps


def append_model_fps(model_path, fps, csv_path="all_fps.csv"):
    model_name = os.path.basename(os.path.normpath(model_path))
    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([model_name, fps])


def iter_model_paths(models_path):
    if os.path.isdir(os.path.join(models_path, "point_cloud")) and os.path.isfile(
        os.path.join(models_path, "cameras.json")
    ):
        yield models_path
        return

    for entry in os.scandir(models_path):
        if entry.is_dir():
            yield entry.path


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        models_path = sys.argv[1]
    else:
        models_path = "/home/kenneth/Documents/3dgs_models/models"  # https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip

    for model_path in iter_model_paths(models_path):
        average_fps = benchmark_model(model_path)
        print(f"{model_path}: {average_fps}\n")
        append_model_fps(model_path, average_fps)
