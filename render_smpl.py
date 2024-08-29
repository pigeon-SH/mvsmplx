import pytorch3d
import pyrender
import trimesh
import numpy as np
from PIL import Image
import os
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesVertex
)
from pytorch3d.utils import cameras_from_opencv_projection

mesh_ids = list(range(20, 150))
cam_ids = [4, 16, 28, 40, 52, 64, 76, 88]

cam_path = "/home/vclab/dataset/Hi4D/talk/talk01/cameras/rgb_cameras.npz"
cam_data = np.load(cam_path)
RTs = cam_data["extrinsics"]
Ks = cam_data["intrinsics"]
device = torch.device("cuda:0")
H, W = 1280, 940

for mesh_id in mesh_ids:
    for cam_idx, cam_id in enumerate(cam_ids):
        RT = torch.Tensor(RTs[cam_idx]).to(device=device)
        K = torch.Tensor(Ks[cam_idx][None]).to(device=device)
        R = RT[:3, :3][None]
        T = RT[:3, 3][None]
        HW = torch.tensor([H, W])[None].to(device=device)
        camera = cameras_from_opencv_projection(R=R, tvec=T, camera_matrix=K, image_size=HW).to(device=device)
        rasterizer = MeshRasterizer(
            cameras=camera,
            raster_settings=RasterizationSettings(
                image_size=(H, W),
                blur_radius=0.0,
                faces_per_pixel=1,
            )
        )
        # lights = PointLights(device=device, location=[[0.0, 2.0, -3.0]])
        lights = PointLights(device=device, location=[[0.0, 2.0, 3.0]])
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=SoftPhongShader(
                device=device, 
                cameras=camera,
                lights=lights
            )
        )
        verts = []
        faces = []
        for person_id in range(2):
            mesh_path = f"/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result_0825_filter_rank/{mesh_id:06d}/smpl_mesh_{person_id}.obj"
            if os.path.exists(mesh_path) == False:
                continue
            vert, face, _ = load_obj(mesh_path)
            face = face.verts_idx
            verts.append(vert)
            faces.append(face)
        faces[1] += len(verts[0])
        verts = torch.concat(verts)
        faces = torch.concat(faces)
        meshes = Meshes(verts=[verts], faces=[faces]).to(device=device)
        verts_rgb = torch.ones_like(verts[None]) * 0.5
        meshes.textures = Textures(verts_rgb=verts_rgb.to(device=device))# torch.ones_like(verts)[None]
        image = renderer(meshes).squeeze()[..., :3]
        img = image.cpu().detach().numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)

        save_path = f"/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result_0825_filter_rank/viz/{mesh_id:06d}_{cam_id:02d}.png"
        img.save(save_path)
