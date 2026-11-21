
import os
import trimesh as tm
import torch
import pytorch3d.structures
import pytorch3d.ops
import numpy as np

from torchsdf import index_vertices_by_faces, compute_sdf
from urdfpy import Mesh, URDF 
from pytorch3d.structures import join_meshes_as_batch, Pointclouds, Meshes
#from pytorch3d.vis.plotly_vis  import plot_scene
import kaolin as kal
from tqdm import tqdm

cache = {}
def load_urdf(path, only_first = False):
    global cache

    if path not in cache:
        urdf = URDF.load(path)
        meshes = []

        for link in urdf.link_fk():
            for visual in link.visuals:
                for mesh in visual.geometry.meshes:
                    verts, faces = torch.from_numpy(mesh.vertices).float(), torch.from_numpy(mesh.faces)
                    if len(verts) > 0 and len(faces) > 0:
                        meshes.append(Meshes(verts=[verts], faces=[faces]))
        meshes = join_meshes_as_batch(meshes)
        cache[path] = meshes
    return cache[path]
    # if '.urdf' in object_path:
    #     urdf = URDF.load(object_path)
    #     fk = urdf.collision_trimesh_fk()
    # else:
    #     object_mesh = tm.load(object_path, force="mesh", process=False)
    
# args: (V, 3), (F, 3)
# ret: (N, 3), (N, 3)
def compute_obj_pc(verts, faces, num_samples):
    face_verts = kal.ops.mesh.index_vertices_by_faces(verts.unsqueeze(0), faces)
    face_normals = kal.ops.mesh.face_normals(face_verts)
    face_features = face_normals.unsqueeze(2).repeat(1, 1, 3, 1)
    pc_verts, pc_faces, pc_normals = kal.ops.mesh.sample_points(verts.unsqueeze(0), faces, num_samples=num_samples, face_features=face_features)
    
    return pc_verts.squeeze(0), pc_normals.squeeze(0)

def subsample_obj_pc(pc_verts, pc_norms, num_samples):
    _, pc_idx = pytorch3d.ops.sample_farthest_points(pc_verts.unsqueeze(0), K=num_samples)
    pc_verts = torch.gather(pc_verts.unsqueeze(0), 1, pc_idx.unsqueeze(-1).repeat(1,1,3))
    pc_norms = torch.gather(pc_norms.unsqueeze(0), 1, pc_idx.unsqueeze(-1).repeat(1,1,3))
    return pc_verts, pc_norms

# args: (B, N, 3), (B, N, 3), (B, N, 3), (B, M, 3)
# ret: (B, M), (B, M, 3), (B, M, 3)
def compute_query_sdfs(pcs_verts, pcs_normals, queries):
    device = pcs_verts.device
    q_dist, q_idx = kal.metrics.pointcloud.sided_distance(queries.cuda(), pcs_verts.cuda())
    q_dist, q_idx = q_dist.to(device), q_idx.to(device)
    counts = torch.zeros(q_idx.shape[0:2], dtype=torch.long, device=device)
    q_norm = torch.gather(pcs_normals, 1, q_idx.unsqueeze(-1).repeat(1,1,3))
    q_closest = torch.gather(pcs_verts, 1, q_idx.unsqueeze(-1).repeat(1,1,3))    
    return q_dist, q_norm, q_closest

def initialize_sdfs(object_paths, num_samples, device):
    """
    Initialize Object Model with list of objects
    
    Choose scales, load meshes, sample surface points
    
    Parameters
    ----------
    object_paths: list | str
        list of object codes
    """

    object_face_verts_list = []
    obj_faces = []
    obj_verts = []
    pcs_verts = []
    pcs_norms = []


    #object_paths = [path.replace('objects', 'urdf').replace('.xml', '.urdf') for path in object_paths if 'urdf' not in path]
    #object_paths = [path.replace('objects', 'urdf').replace('.xml', '.urdf') if 'urdf' not in path else path for path in object_paths]
    for object_path in tqdm(object_paths, "Loading SDF"):
        # object_verts = torch.tensor(object_mesh.vertices, dtype=torch.float, device=device)
        # object_faces = torch.tensor(object_mesh.faces, dtype=torch.long, device=device)
        # obj_verts.append(object_verts)
        # obj_faces.append(object_faces)
        # mesh = pytorch3d.structures.Meshes(object_verts.unsqueeze(0), object_faces.unsqueeze(0))

        mesh = load_urdf(object_path, only_first=True)
        object_verts = mesh.verts_packed().to(device=device)
        object_faces = mesh.faces_packed().to(device=device)
        #object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))

        if False:
            pc_verts, pc_norms = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples, return_normals=True)
            pc_verts, pc_norms = pc_verts.squeeze(0), pc_norms.squeeze(0)
        else:
            pc_verts, pc_norms = compute_obj_pc(object_verts, object_faces, num_samples=50 * num_samples)
        pc_verts, pc_norms = pc_verts.to(device), pc_norms.to(device)
        pc_norms = pc_norms / pc_norms.norm(dim=-1, keepdim=True)
        pc_verts, pc_norms = subsample_obj_pc(pc_verts, pc_norms, num_samples)
        pcs_verts.append(pc_verts.squeeze(0))
        pcs_norms.append(pc_norms.squeeze(0))

    #return obj_verts, obj_faces, object_face_verts_list, 
    return torch.stack(pcs_verts), torch.stack(pcs_norms)


def compute_distance_sdfs(x, object_face_verts_list, with_closest_points=False):
    """
    Calculate signed distances from hand contact points to object meshes and return contact normals
    
    Interiors are positive, exteriors are negative
    
    Use our modified Kaolin package
    
    Parameters
    ----------
    x: (B, `n_contact`, 3) torch.Tensor
        hand contact points
    with_closest_points: bool
        whether to return closest points on object meshes
    
    Returns
    -------
    distance: (B, `n_contact`) torch.Tensor
        signed distances from hand contact points to object meshes, inside is positive
    normals: (B, `n_contact`, 3) torch.Tensor
        contact normal vectors defined by gradient
    closest_points: (B, `n_contact`, 3) torch.Tensor
        contact points on object meshes, returned only when `with_closest_points is True`
    """
    n_meshes, n_points, _ = x.shape
    assert(n_meshes == len(object_face_verts_list))
    distance = []
    normals = []
    closest_points = []
    for i in range(n_meshes):
        face_verts = object_face_verts_list[i]
        dis, dis_signs, normal, _ = compute_sdf(x[i].contiguous(), face_verts.contiguous())
        if with_closest_points:
            closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
        dis = torch.sqrt(dis + 1e-8)
        dis = dis * (-dis_signs)
        distance.append(dis)
        normals.append(normal * dis_signs.unsqueeze(1))
    
    distance = torch.stack(distance)
    normals = torch.stack(normals)
    
    if with_closest_points:
        closest_points = torch.stack(closest_points)
        return distance, normals, closest_points
    return distance, normals


if __name__ == '__main__':
    meshes = load_urdf('../assets/pregrasp_assets/assets/urdf/banana.urdf')
    fig = plot_scene({
        "subplot_title": {
            "mesh": meshes,
        }
    })
    fig.show()

if __name__ == '__main__':
    object_paths = [
        '/home/akirchme/catch/grogu/DyanmicTasks/assets/pregrasp_assets/assets/objects/banana.xml'
    ]

    base_names = [os.path.basename(path) for path in object_paths]
    object_paths = [path.replace('assets/objects', 'assets/meshes/objects').replace('.xml', f'/{base_name.split(".")[0]}.stl') for base_name, path in zip(base_names, object_paths)]
    num_samples = 100
    device = 'cuda'

    obj_verts, obj_faces, obj_vert_faces, surface_points = initialize(object_paths, num_samples, device)
    print(surface_points.shape)
    obj_scales = torch.ones(surface_points.shape[0], dtype=torch.float, device=device)
    obj_positions = torch.zeros((surface_points.shape[0], 3), dtype=torch.float, device=device)
    queries = torch.zeros((surface_points.shape[0], 1, 3), dtype=torch.float, device=device)
    obj_vert_faces = [obj*obj_scale + obj_pos.unsqueeze(0) for obj, obj_scale, obj_pos in zip(obj_vert_faces, obj_scales, obj_positions)]
    distance, normals, closest_points = compute_distance_sdfs(queries, obj_vert_faces, with_closest_points=True)
    print(distance.shape, normals.shape, closest_points.shape)
    # Visualization

    from pytorch3d.vis.plotly_vis  import plot_scene
    from pytorch3d.structures import Pointclouds, Meshes
    fig = plot_scene({
        "subplot_title": {
            #"mesh": Meshes(verts=obj_verts, faces=obj_faces),
            "surface":  Pointclouds(points=surface_points),
            "queries":  Pointclouds(points=queries.reshape(1, -1, 3)),
            "closest_points":  Pointclouds(points=closest_points.reshape(1, -1, 3))
        }
    })
    fig.show()
