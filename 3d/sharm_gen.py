import trimesh
import math as m
import numpy as np
import pyshtools
import glob
import os

def change_coordinates(coords, p_from='C', p_to='S'):
    if p_from == p_to:
        return coords
    elif p_from == 'S' and p_to == 'C':

        beta = coords[..., 0]
        alpha = coords[..., 1]
        r = 1.

        out = np.empty(beta.shape + (3,))

        ct = np.cos(beta)
        cp = np.cos(alpha)
        st = np.sin(beta)
        sp = np.sin(alpha)
        out[..., 0] = r * st * cp  # x
        out[..., 1] = r * st * sp  # y
        out[..., 2] = r * ct       # z
        return out

    elif p_from == 'C' and p_to == 'S':

        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]

        out = np.empty(x.shape + (2,))
        out[..., 0] = np.arccos(z)         # beta
        out[..., 1] = np.arctan2(y, x)     # alpha
        return out

    else:
        raise ValueError('Unknown conversion:' + str(p_from) + ' to ' + str(p_to))

def make_sgrid_(b):
    theta = np.linspace(0, m.pi, num=b)
    phi = np.linspace(0, 2*m.pi, num=b)
    theta_m, phi_m = np.meshgrid(theta, phi)
    sgrid = change_coordinates(np.c_[theta_m[..., None], phi_m[..., None]], p_from='S', p_to='C')
    sgrid = sgrid.reshape((-1, 3))


    return sgrid

def render_model(mesh, sgrid):

    # Cast rays
    # triangle_indices = mesh.ray.intersects_first(ray_origins=sgrid, ray_directions=-sgrid)
    s_or = np.zeros(sgrid.shape)
    index_tri, index_ray, loc = mesh.ray.intersects_id(
        ray_origins=-sgrid, ray_directions=sgrid, multiple_hits=False, return_locations=True)
    loc = loc.reshape((-1, 3)) # fix bug if loc is empty
    final_loc = np.zeros((sgrid.shape[0],3))
    final_loc[index_ray] = loc

    return final_loc

def cart2sph(coords):
        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]

        r = np.sqrt(np.power(x,2) + np.power(y,2) + np.power(z,2))
        xy = np.sqrt(np.power(x,2) + np.power(y,2))
        print(x.shape)
        print(r.shape)
        out = np.empty(x.shape + (3,))
        out[..., 0] = np.arctan2(xy,z)
        out[..., 1] = np.arctan2(y, x)

       # out[..., 0] = np.arccos(z)         # beta
       # out[..., 1] = np.arctan2(y, x)     # alpha
        out[..., 2] = np.sqrt(np.power(x,2) + np.power(y,2) + np.power(z,2))
#       print(out)
        return out

def sph2cart(coords):
        beta = coords[..., 0]
        alpha = coords[..., 1]
        r = coords[..., 2]

        out = np.empty(beta.shape + (3,))

        ct = np.cos(beta)
        cp = np.cos(alpha)
        st = np.sin(beta)
        sp = np.sin(alpha)
        out[..., 0] = r * st * cp  # x
        out[..., 1] = r * st * sp  # y
        out[..., 2] = r * ct       # z
        return out

file_list = glob.glob(os.path.join('/flush5/ram095/nips20/datasets/shapenet/ShapeNetCore.v2.zip/', '*.obj'))
sgrid = make_sgrid_(128)
itr=0
mitr = 0
nPoints = 0
for filename in file_list:
    try:
        if 'chair' in filename:
                mesh = trimesh.load(filename)
                mitr = mitr + 1
                print(filename)
                print(mesh.vertices.shape[0])
                if mesh.vertices.shape[0] < 15000:
                        if itr > 0:
                                print(mesh.vertices.shape)
                                mesh.remove_degenerate_faces()
                                mesh.fix_normals()
                                mesh.fill_holes()
                                mesh.remove_duplicate_faces()
                                mesh.remove_infinite_values()
                                mesh.remove_unreferenced_vertices()
                                mesh.apply_translation(-mesh.centroid)
                                r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
                                mesh.apply_scale(1 / r)
                                loc = render_model(mesh,sgrid)
                                k = cart2sph(loc)
                                r_ = k[:,2].reshape(128,128)
                                print("generating harmonics for" + filename)
                                cilm = pyshtools.expand.SHExpandDH(r_)
                                np.save('/scratch1/ram095/nips20/datasets/sharm/' + str(itr)+".npy", cilm)
                                print(cilm.shape)
        #               np.save('/media/ram095/329CCC2B9CCBE785/samples/' + str(itr)+".npy", k)
                        itr= itr+1

    except:
        print("skipped")


