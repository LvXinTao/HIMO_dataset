"""
This script contains functions for converting between different rotation representations.
Including:
    - Euler angles (euler) default to (XYZ,intrinsic)
    - Rotation matrices (rot)
    - Quaternions (quat)
    - Axis-angle (aa)
    - 6D representation (6d)
We also provide numpy and torch versions of these functions.
Note that all functions are in batch mode.
"""
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms.rotation_conversions import (
    quaternion_to_axis_angle,quaternion_to_matrix,
    axis_angle_to_matrix,axis_angle_to_quaternion,
    matrix_to_axis_angle,matrix_to_quaternion,
    euler_angles_to_matrix,matrix_to_euler_angles,
    rotation_6d_to_matrix,matrix_to_rotation_6d
)

#--------------------Numpy Version--------------------------#
def euler2rot_numpy(euler,degrees=False):
    """
    euler:[B,3] (XYZ,extrinsic)
    degrees are False if they are radians
    return: [B,3,3]
    """
    assert isinstance(euler, np.ndarray)
    ori_shape = euler.shape[:-1]
    rots = np.reshape(euler, (-1, 3))
    rots = R.as_matrix(R.from_euler("XYZ", rots, degrees=degrees))
    rot = np.reshape(rots, ori_shape + (3, 3))
    return rot

def rot2euler_numpy(rot,degrees=False):
    """
    rot:[B,3,3]
    return: [B,3]
    """
    assert isinstance(rot, np.ndarray)
    ori_shape = rot.shape[:-2]
    rots = np.reshape(rot, (-1, 3, 3))
    rots = R.as_euler(R.from_matrix(rots), "XYZ", degrees=degrees)
    euler = np.reshape(rots, ori_shape + (3,))
    return euler

def euler2quat_numpy(euler,degrees=False):
    """
    euler:[B,3]
    return [B,4]
    """
    assert isinstance(euler,np.ndarray)
    ori_shape=euler.shape[:-1]
    rots=np.reshape(euler,(-1,3))
    quats=R.as_quat(R.from_euler("XYZ",rots,degrees=degrees))
    quat=np.reshape(quats,ori_shape+(4,))
    return quat

def quat2euler_numpy(quat,degrees=False):
    """
    quat:[B,4]
    return [B,3]
    """
    assert isinstance(quat,np.ndarray)
    ori_shape=quat.shape[:-1]
    rots=np.reshape(quat,(-1,3))
    eulers=R.as_euler("XYZ",R.from_quat(rots),degrees=degrees)
    euler=np.reshape(eulers,ori_shape+(3,))
    return euler

def euler2aa_numpy(euler,degrees=False):
    """
    euler:[B,3]
    return: [B,3]
    """
    assert isinstance(euler, np.ndarray)
    ori_shape = euler.shape[:-1]
    rots = np.reshape(euler, (-1, 3))
    aas = R.as_rotvec(R.from_euler("XYZ", rots, degrees=degrees))
    rotation_vectors = np.reshape(aas, ori_shape + (3,))
    return rotation_vectors

def aa2euler_numpy(aa,degrees=False):
    """
    aa:[B,3]
    return [B,3]
    """
    assert isinstance(aa, np.ndarray)
    ori_shape = aa.shape[:-1]
    aas = np.reshape(aa, (-1, 3))
    rots = R.as_euler(R.from_rotvec(aas), "XYZ", degrees=degrees)
    euler_angles = np.reshape(rots, ori_shape + (3,))
    return euler_angles

def rot2quat_numpy(rot):
    """
    rot:[B,3,3]
    return [B,4]
    """
    return euler2quat_numpy(rot2euler_numpy(rot))

def quat2rot_numpy(quat):
    """
    quat:[B,4] (w,x,y,z)
    return: [B,3,3]
    """
    return euler2rot_numpy(quat2euler_numpy(quat))

def rot2aa_numpy(rot):
    """
    rot:[B,3,3]
    return:[B,3]
    """
    assert isinstance(rot, np.ndarray)
    ori_shape = rot.shape[:-2]
    rots = np.reshape(rot, (-1, 3, 3))
    aas = R.as_rotvec(R.from_matrix(rots))
    rotation_vectors = np.reshape(aas, ori_shape + (3,))
    return rotation_vectors

def aa2rot_numpy(aa):
    """
    aa:[B,3]
    Rodirgues formula
    return: [B,3,3]
    """
    assert isinstance(aa,np.ndarray)
    ori_shape=aa.shape[:-1]
    aas=np.reshape(aa,(-1,3))
    rots=R.as_matrix(R.from_rotvec(aas))
    rot_mat=np.reshape(rots,ori_shape+(3,3))
    return rot_mat

def quat2aa_numpy(quat):
    """
    quat:[B,4]
    return [B,3]
    """
    return euler2aa_numpy(quat2euler_numpy(quat))

def aa2quat_numpy(aa):
    """
    aa:[B,3]
    return [B,4]
    """
    return euler2quat_numpy(aa2euler_numpy(aa))

def rot2sixd_numpy(rot):
    """
    rot:[B,3,3]
    return [B,6]
    """
    assert isinstance(rot,np.ndarray)
    ori_shape=rot.shape[:-2]
    return rot[...,:2,:].copy().reshape(ori_shape+(6,))

def sixd2rot_numpy(sixd):
    """
    sixd:[B,6]
    return [B,3,3]
    """
    assert isinstance(sixd,np.ndarray)
    a1,a2=sixd[...,:3],sixd[...,3:]
    b1=a1/np.linalg.norm(a1,axis=-1,keepdims=True)
    b2=a2-(b1*a2).sum(-1,keepdims=True)*b1
    b2=b2/np.linalg.norm(b2,axis=-1,keepdims=True)
    b3=np.cross(b1,b2,axis=-1)
    return np.stack([b1,b2,b3],axis=-2)

def rpy2rot_numpy(rpy,degrees=False):
    """
    rpy: [B,3] (ZYX,intrinsic)
    return [B,3,3]
    """
    assert isinstance(rpy, np.ndarray)
    ori_shape = rpy.shape[:-1]
    rots = np.reshape(rpy, (-1, 3))
    rots = R.as_matrix(R.from_euler("ZYX", rots, degrees=degrees))
    rotation_matrices = np.reshape(rots, ori_shape + (3, 3))
    return rotation_matrices

#--------------------Pytorch Version--------------------------#
def euler2rot_torch(euler,degrees=False):
    """
    euler [B,3] (XYZ,intrinsic)
    degrees are False if they are radians
    """
    if degrees:
        euler_rad=torch.deg2rad(euler)
        return euler_angles_to_matrix(euler_rad,"XYZ")
    else:
        return euler_angles_to_matrix(euler,"XYZ")
    
def rot2euler_torch(rot,degrees=False):
    """
    rot:[B,3,3]
    return: [B,3]
    """
    if degrees:
        euler_rad=matrix_to_euler_angles(rot,"XYZ")
        return torch.rad2deg(euler_rad)
    else:
        return matrix_to_euler_angles(rot,"XYZ")
    
def euler2quat_torch(euler,degrees=False):
    """
    euler:[B,3]
    return [B,4]
    """
    if degrees:
        euler_rad=torch.deg2rad(euler)
        return matrix_to_quaternion(euler_angles_to_matrix(euler_rad,"XYZ"))
    else:
        return matrix_to_quaternion(euler_angles_to_matrix(euler,"XYZ"))

def quat2euler_torch(quat,degrees=False):
    """
    quat:[B,4]
    return [B,3]
    """
    if degrees:
        euler_rad=quaternion_to_matrix(quat)
        return torch.rad2deg(matrix_to_euler_angles(euler_rad,"XYZ"))
    else:
        return matrix_to_euler_angles(quaternion_to_matrix(quat),"XYZ")

def euler2aa_torch(euler,degrees=False):
    """
    euler:[B,3]
    return: [B,3]
    """
    if degrees:
        euler_rad=torch.deg2rad(euler)
        return matrix_to_axis_angle(euler_angles_to_matrix(euler_rad,"XYZ"))
    else:
        return matrix_to_axis_angle(euler_angles_to_matrix(euler,"XYZ"))

def aa2euler_torch(aa,degrees=False):
    """
    aa:[B,3]
    return [B,3]
    """
    if degrees:
        euler_rad=axis_angle_to_matrix(aa)
        return torch.rad2deg(matrix_to_euler_angles(euler_rad,"XYZ"))
    else:
        return matrix_to_euler_angles(axis_angle_to_matrix(aa),"XYZ")

def rot2quat_torch(rot):
    """
    rot:[B,3,3]
    return [B,4]
    """
    return matrix_to_quaternion(rot)

def quat2rot_torch(quat):
    """
    quat:[B,4] (w,x,y,z)
    return: [B,3,3]
    """
    return quaternion_to_matrix(quat)

def rot2aa_torch(rot):
    """
    rot:[B,3,3]
    return:[B,3]
    """
    return matrix_to_axis_angle(rot)

def aa2rot_torch(aa):
    """
    aa:[B,3]
    Rodirgues formula
    return: [B,3,3]
    """
    return axis_angle_to_matrix(aa)

def quat2aa_torch(quat):
    """
    quat:[B,4]
    return [B,3]
    """
    return quaternion_to_axis_angle(quat)

def aa2quat_torch(aa):
    """
    aa:[B,3]
    return [B,4]
    """
    return axis_angle_to_quaternion(aa)

def rot2sixd_torch(rot):
    """
    rot:[B,3,3]
    return [B,6]
    """
    return matrix_to_rotation_6d(rot)

def sixd2rot_torch(sixd):
    """
    sixd:[B,6]
    return [B,3,3]
    """
    return rotation_6d_to_matrix(sixd)

if __name__=='__main__':
    rot=np.eye(3)[None,...].repeat(2,axis=0)
    sixd=rot2sixd_numpy(rot)
    new_rot=sixd2rot_numpy(sixd)
    print(new_rot)