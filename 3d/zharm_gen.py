import trimesh
import math as m

import numpy as np

import glob
import os
import scipy


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
    print(sgrid.shape)
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
        out[..., 1] = np.arctan2(y, x) + m.pi

       # out[..., 0] = np.arccos(z)         # beta
       # out[..., 1] = np.arctan2(y, x)     # alpha
        out[..., 2] = np.sqrt(np.power(x,2) + np.power(y,2) + np.power(z,2))
#       print(out)
        return out


def radial_poly(rho, m, n):
        if n == 0 and m == 0:
                return np.ones(rho.shape)
        if n == 1 and m == 1:
                return rho
        if n == 2 and m == 0:
                return 2.0 * np.power(rho,2) - 1
        if n == 2 and m == 2:
                return np.power(rho,2)
        if n == 3 and m == 1:
                return 3.0* np.power(rho, 3) - 2.0 * rho
        if n == 3 and m == 3:
                return np.power(rho,3)
        if n == 4 and m == 0:
                return 6.0 * np.power(rho,4) - 6.0 * np.power(rho,2) + 1
        if n == 4 and m == 2:
                return 4.0* np.power(rho, 4) - 3.0 * np.power(rho,2)
        if n == 4 and m == 4:
                return np.power(rho,4)
        if n == 5 and m == 1:
                return 10.0* np.power(rho, 5) - 12.0 * np.power(rho, 3) + 3.0 * rho
        if n == 5 and m == 3:
                return 5.0 * np.power(rho, 5) - 4.0 * np.power(rho, 3)
        if n == 5 and m == 5:
                return np.power(rho,5)
        if n == 6 and m == 0:
                return 20.0 * np.power(rho, 6) - 30.0 * np.power(rho, 4) + 12.0 * np.power(rho, 2) -1
        if n == 6 and m == 2:
                return 15.0* np.power(rho, 6) - 20.0 *  np.power(rho, 4) + 6.0 * np.power(rho,2)
        if n == 6 and m == 4:
                return 6.0 * np.power(rho, 6)  - 5.0 * np.power(rho,4)
        if n == 6 and m == 6:
                return np.power(rho, 6)

def spherical_harmonic(m, n, theta, phi):
        return scipy.special.sph_harm(m, n, theta, phi)

def X_gen(r, theta, phi):
        r = np.clip(r, 0, 1.0)
        u_0_0_0 = spherical_harmonic(0,0,theta,phi) * radial_poly(r, 0, 0)
        u_0_1_1 = spherical_harmonic(0,1,theta,phi) * radial_poly(r, 1, 1)
        u_1_1_1 = spherical_harmonic(1,1,theta,phi) * radial_poly(r, 1, 1)
        u_0_0_2 = spherical_harmonic(0,2,theta,phi) * radial_poly(r, 0, 2)
        u_0_2_2 = spherical_harmonic(0,2,theta,phi) * radial_poly(r, 2, 2)
        u_1_2_2 = spherical_harmonic(1,2,theta,phi) * radial_poly(r, 2, 2)
        u_2_2_2 = spherical_harmonic(2,2,theta,phi) * radial_poly(r, 2, 2)
        u_0_1_3 = spherical_harmonic(0,1,theta,phi) * radial_poly(r, 1, 3)
        u_1_1_3 = spherical_harmonic(1,1,theta,phi) * radial_poly(r, 1, 3)
        u_0_3_3 = spherical_harmonic(0,3,theta,phi) * radial_poly(r, 3, 3)
        u_1_3_3 = spherical_harmonic(1,3,theta,phi) * radial_poly(r, 3, 3)
        u_2_3_3 = spherical_harmonic(2,3,theta,phi) * radial_poly(r, 3, 3)
        u_3_3_3 = spherical_harmonic(3,3,theta,phi) * radial_poly(r, 3, 3)
        u_0_0_4 = spherical_harmonic(0,0,theta,phi) * radial_poly(r, 0, 4)
        u_0_2_4 = spherical_harmonic(0,2,theta,phi) * radial_poly(r, 2, 4)
        u_1_2_4 = spherical_harmonic(1,2,theta,phi) * radial_poly(r, 2, 4)
        u_2_2_4 = spherical_harmonic(2,2,theta,phi) * radial_poly(r, 2, 4)
        u_0_4_4 = spherical_harmonic(0,4,theta,phi) * radial_poly(r, 4, 4)
        u_1_4_4 = spherical_harmonic(1,4,theta,phi) * radial_poly(r, 4, 4)
        u_2_4_4 = spherical_harmonic(2,4,theta,phi) * radial_poly(r, 4, 4)
        u_3_4_4 = spherical_harmonic(3,4,theta,phi) * radial_poly(r, 4, 4)
        u_4_4_4 = spherical_harmonic(4,4,theta,phi) * radial_poly(r, 4, 4)
        u_0_1_5 = spherical_harmonic(0,1,theta,phi) * radial_poly(r, 1, 5)
        u_1_1_5 = spherical_harmonic(1,1,theta,phi) * radial_poly(r, 1, 5)
        u_0_3_5 = spherical_harmonic(0,3,theta,phi) * radial_poly(r, 3, 5)
        u_1_3_5 = spherical_harmonic(1,3,theta,phi) * radial_poly(r, 3, 5)
        u_2_3_5 = spherical_harmonic(2,3,theta,phi) * radial_poly(r, 3, 5)
        u_3_3_5 = spherical_harmonic(3,3,theta,phi) * radial_poly(r, 3, 5)
        u_0_5_5 = spherical_harmonic(0,5,theta,phi) * radial_poly(r, 5, 5)
        u_1_5_5 = spherical_harmonic(1,5,theta,phi) * radial_poly(r, 5, 5)
        u_2_5_5 = spherical_harmonic(2,5,theta,phi) * radial_poly(r, 5, 5)
        u_3_5_5 = spherical_harmonic(3,5,theta,phi) * radial_poly(r, 5, 5)
        u_4_5_5 = spherical_harmonic(4,5,theta,phi) * radial_poly(r, 5, 5)
        u_5_5_5 = spherical_harmonic(5,5,theta,phi) * radial_poly(r, 5, 5)
        u_0_0_6 = spherical_harmonic(0,0,theta,phi) * radial_poly(r, 0, 6)
        u_0_2_6 = spherical_harmonic(0,2,theta,phi) * radial_poly(r, 2, 6)
        u_1_2_6 = spherical_harmonic(1,2,theta,phi) * radial_poly(r, 2, 6)
        u_2_2_6 = spherical_harmonic(2,2,theta,phi) * radial_poly(r, 2, 6)
        u_0_4_6 = spherical_harmonic(0,4,theta,phi) * radial_poly(r, 4, 6)
        u_1_4_6 = spherical_harmonic(1,4,theta,phi) * radial_poly(r, 4, 6)
        u_2_4_6 = spherical_harmonic(2,4,theta,phi) * radial_poly(r, 4, 6)
        u_3_4_6 = spherical_harmonic(3,4,theta,phi) * radial_poly(r, 4, 6)
        u_4_4_6 = spherical_harmonic(4,4,theta,phi) * radial_poly(r, 4, 6)
        u_0_6_6 = spherical_harmonic(0,6,theta,phi) * radial_poly(r, 6, 6)
        u_1_6_6 = spherical_harmonic(1,6,theta,phi) * radial_poly(r, 6, 6)
        u_2_6_6 = spherical_harmonic(2,6,theta,phi) * radial_poly(r, 6, 6)
        u_3_6_6 = spherical_harmonic(3,6,theta,phi) * radial_poly(r, 6, 6)
        u_4_6_6 = spherical_harmonic(4,6,theta,phi) * radial_poly(r, 6, 6)
        u_5_6_6 = spherical_harmonic(5,6,theta,phi) * radial_poly(r, 6, 6)
        u_6_6_6 = spherical_harmonic(6,6,theta,phi) * radial_poly(r, 6, 6)

        U = np.real(np.concatenate([u_0_0_0, u_0_1_1, u_1_1_1, u_0_0_2, u_0_2_2, u_1_2_2, u_2_2_2,
                       u_0_1_3, u_1_1_3, u_0_3_3, u_1_3_3, u_2_3_3, u_3_3_3,
                       u_0_0_4, u_0_2_4, u_1_2_4, u_2_2_4, u_0_4_4, u_1_4_4, u_2_4_4, u_3_4_4, u_4_4_4, u_0_1_5,
                       u_1_1_5, u_0_3_5, u_1_3_5, u_2_3_5, u_3_3_5, u_0_5_5, u_1_5_5, u_2_5_5, u_3_5_5, u_4_5_5,
                       u_5_5_5, u_0_0_6, u_0_2_6, u_1_2_6, u_2_2_6, u_0_4_6, u_1_4_6, u_2_4_6, u_3_4_6, u_4_4_6,
                       u_0_6_6, u_1_6_6, u_2_6_6, u_3_6_6, u_4_6_6, u_5_6_6, u_6_6_6], axis=1))

        V = np.imag(np.concatenate([u_0_0_0, u_0_1_1, u_1_1_1, u_0_0_2, u_0_2_2, u_1_2_2, u_2_2_2,
                                    u_0_1_3, u_1_1_3, u_0_3_3, u_1_3_3, u_2_3_3, u_3_3_3,
                                    u_0_0_4, u_0_2_4, u_1_2_4, u_2_2_4, u_0_4_4, u_1_4_4, u_2_4_4, u_3_4_4, u_4_4_4,
                                    u_0_1_5,
                                    u_1_1_5, u_0_3_5, u_1_3_5, u_2_3_5, u_3_3_5, u_0_5_5, u_1_5_5, u_2_5_5, u_3_5_5,
                                    u_4_5_5,
                                    u_5_5_5, u_0_0_6, u_0_2_6, u_1_2_6, u_2_2_6, u_0_4_6, u_1_4_6, u_2_4_6, u_3_4_6,
                                    u_4_4_6,
                                    u_0_6_6, u_1_6_6, u_2_6_6, u_3_6_6, u_4_6_6, u_5_6_6, u_6_6_6], axis=1))

        X = np.concatenate([U, V], axis=1)
        return X

def create_zpol(points):
        r = np.reshape(points[:,2],(-1,1))
#theta = np.reshape(points[:,1],(-1,1))
        theta_= np.linspace(0, 2* m.pi, num=50)
        phi_= np.linspace(0, m.pi, num=50)
        theta, phi = np.meshgrid(theta_, phi_)
#phi = np.reshape(points[:,0],(-1,1))
        theta= np.reshape(theta,(-1,1))
        phi= np.reshape(phi,(-1,1))
        X = X_gen(r,theta,phi)
        X_inv = np.linalg.pinv(X)

        C = np.matmul(X_inv,r)
        return C

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

file_list = sorted(glob.glob(os.path.join('/flush5/ram095/nips20/datasets/ModelNet10/bathtub/test/', '*.off')))
sgrid = make_sgrid_(50)
print(sgrid.shape)
itr=1
mitr = 0
nPoints = 0
for filename in file_list:
    try:
        if True: #'bath' in filename:
                  print(filename)
                  if itr > 0:
                        mesh = trimesh.load(filename)
#                mitr = mitr + 1
                        #print(filename)
                        print("number of vertices: "+ str( mesh.vertices.shape[0]))
                        if mesh.vertices.shape[0] < 20000:
                        #if itr > 24524:
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
                                print(sgrid.shape)
                                loc = render_model(mesh,sgrid)
                                k = cart2sph(loc)
                                C = create_zpol(k)
                                print(np.max(C))
                                print(np.min(C))
                                base_name =  os.path.splitext(os.path.basename(filename))[0]
                                print(base_name)
                                print("generating harmonics for" + filename)
                                print( str(itr) +".npy")
        #       print(cilm)     print(" 
                                np.save('/scratch1/ram095/nips20/datasets/ModelNet10harm/test/bath/' + base_name +".npy", C)

        #               np.save('/media/ram095/329CCC2B9CCBE785/samples/' + str(itr)+".npy", k)
                  itr= itr+1
    except:
        print("skipped")


