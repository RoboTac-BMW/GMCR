import numpy as np
from fitting_graphs.utility.functions import invert_transform_sep
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
import json
import open3d as o3d
import copy

def to_T(s, R, t):

    T_ = np.zeros((4, 4))

    T_[0:3, 0:3] = s * R
    T_[0:3, 3] = t
    T_[3, 3] = 1
    return T_


class O3DPCDWrapper:

    def __init__(self, o3d_pcd):
        self.o3d_pcd = o3d_pcd
        self.last_R = None
        self.last_t = None
        self.last_s = None
        self.gt_s = None
        self.gt_R = None
        self.gt_t = None
        self.int_res = None
        self.slerp = None

    def create_interpolate(self, steps, s, R, t, only=None):
        self.gt_s = s
        self.gt_R = R
        self.gt_t = t
        self.int_res = 1 / steps
        self.only = only

        R_start = np.identity(3)
        R_end = R
        self.slerp = Slerp([0, 1], Rot.from_matrix([R_start, R_end]))

    def interpolate(self, step):

        # apply inverse of prev_step
        if self.last_s is not None:
            R_inv, t_inv = invert_transform_sep(self.last_R, self.last_t)
            s_inv = 1 / self.last_s
            T_inv = to_T(s_inv, R_inv, s_inv * t_inv)
            self.o3d_pcd.transform(T_inv)
            pcd = np.array(self.o3d_pcd.points).mean()
            print(pcd)

        s_ = 1 * (1 - self.int_res * step) + self.gt_s * (self.int_res * step)
        if self.only == 's' or self.only == 't':
            R_ = np.identity(3)
        else:
            R_ = self.slerp(self.int_res * step).as_matrix()
        if self.only == 's' or self.only == 'R':
            t_ = np.zeros(3)
        else:
            t_ = self.gt_t * step * self.int_res
        T_ = to_T(s_, R_, t_)
        self.o3d_pcd.transform(T_)

        print(s_)
        print(R_)
        print(t_)
        self.last_s = s_
        self.last_R = R_
        self.last_t = t_


def get_window_dimensions_from_file(file_path):

    with open(file_path, 'r') as f:

        content = json.load(f)

    width, height = content['intrinsic']['width'], content['intrinsic']['height']

    return width, height


def animate_registration(samp, viewer_file, steps, out_file, only=None, reg_res=None):

    if reg_res is None:
        s_gt, R_gt, t_gt = samp.s, samp.R, samp.t
    else:
        s_gt, R_gt, t_gt = reg_res

    if only == 'R' or only == 't':
        samp.source = samp.source * s_gt
        s_gt = 1

    if only == 't':
        samp.source.transform(R_gt, np.zeros(3))
        R_gt = np.identity(3)

    source_raw = O3DPCDWrapper(samp.source.to_o3d())
    STEPS = steps
    source_raw.create_interpolate(STEPS - 1, s_gt, R_gt, t_gt, only)
    # samp.target.display()
    target_raw = samp.target.to_o3d()

    # samp.source.transform(samp.s * samp.R, samp.t)
    # samp.display(show_corresp=False)
    width, height = get_window_dimensions_from_file(file_path=viewer_file)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    # Add PCDs
    vis.add_geometry(source_raw.o3d_pcd)
    vis.add_geometry(target_raw)
    # Add lines
    c_inlier = copy.deepcopy(samp.C_true)
    c_outlier = copy.deepcopy(samp.C_false)
    line_points = np.concatenate([samp.source.points, samp.target.points], axis=0)
    line_ids = c_inlier
    line_ids[:, 1] += samp.source.points.shape[0]

    true_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(line_ids),
    )
    colors = [[0, 1, 0] for i in range(len(line_ids))]
    true_line_set.colors = o3d.utility.Vector3dVector(colors)

    c_outlier[:, 1] += samp.source.points.shape[0]

    false_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(c_outlier),
    )
    colors = [[1, 0, 0] for i in range(len(c_outlier))]
    false_line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(true_line_set)
    vis.add_geometry(false_line_set)

    param = o3d.io.read_pinhole_camera_parameters(viewer_file)
    intrinsic = param.intrinsic.intrinsic_matrix
    extrinsic = param.extrinsic
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
    param.intrinsic.intrinsic_matrix = intrinsic
    param.extrinsic = extrinsic
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    threshold = 0.05
    icp_iteration = 100
    vis.poll_events()
    vis.update_renderer()
    save_image = False
    resolution = 0.01
    imgs = []
    for i in np.arange(STEPS):
        #    print(reg_p2l.transformation)
        #    source.transform(reg_p2l.transformation)
        source_raw.interpolate(i)
        vis.update_geometry(source_raw.o3d_pcd)
        source_points = np.array(source_raw.o3d_pcd.points)
        line_points = np.concatenate([source_points, samp.target.points], axis=0)
        if not STEPS - 2 < i:
            true_line_set.points = o3d.utility.Vector3dVector(line_points)
            false_line_set.points = o3d.utility.Vector3dVector(line_points)
        else:
            true_line_set.points = o3d.utility.Vector3dVector([])
            true_line_set.colors = o3d.utility.Vector3dVector([])
            false_line_set.points = o3d.utility.Vector3dVector([])
            false_line_set.colors = o3d.utility.Vector3dVector([])

        vis.update_geometry(true_line_set)
        vis.update_geometry(false_line_set)
        vis.poll_events()
        vis.update_renderer()
        # time.sleep(0.5)
        img = vis.capture_screen_float_buffer(do_render=True)
        imgs.append(np.array(img))
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)
    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
    imgs = [Image.fromarray((img * 255).astype(np.uint8)) for img in imgs]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(out_file, save_all=True, append_images=imgs[1:], duration=50, loop=0)


