import os
import time

import glfw
import imgui
import numpy as np
import trimesh

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence

glfw.init()
primary_monitor = glfw.get_primary_monitor()
mode = glfw.get_video_mode(primary_monitor)
width = mode.size.width
height = mode.size.height

C.update_conf({'window_width': width*0.9, 'window_height': height*0.9})
C.update_conf({'smplx_models':'./body_models'})

class SMPLX_Viewer(Viewer):
    title='HIMO Viewer for SMPL-X' 

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.gui_controls.update(
            {
                'show_text':self.gui_show_text
            }
        )
        self._set_prev_record=self.wnd.keys.UP
        self._set_next_record=self.wnd.keys.DOWN

        # reset
        self.reset_for_himo()
        self.load_one_sequence()

    def reset_for_himo(self):
        
        self.text_val = ''

        self.clip_folder = os.path.join('data','smplx')
        self.text_folder = os.path.join('data','text')
        self.object_pose_folder = os.path.join('data','object_pose')
        self.object_mesh_folder = os.path.join('data','object_mesh')
        
        # Pre-load object meshes
        self.object_mesh={}
        for obj in os.listdir(self.object_mesh_folder):
            if not obj.startswith('.'):
                obj_name = obj.split('.')[0]
                obj_path = os.path.join(self.object_mesh_folder, obj)
                mesh = trimesh.load(obj_path)
                self.object_mesh[obj_name] = mesh

        self.label_npy_list = []
        self.get_label_file_list()
        self.total_tasks = len(self.label_npy_list)

        self.label_pid = 0
        self.go_to_idx = 0

    def key_event(self, key, action, modifiers):
        if action==self.wnd.keys.ACTION_PRESS:
            if key==self._set_prev_record:
                self.set_prev_record()
            elif key==self._set_next_record:
                self.set_next_record()
            else:
                return super().key_event(key, action, modifiers)
        else:
            return super().key_event(key, action, modifiers)

    def gui_show_text(self):
        imgui.set_next_window_position(self.window_size[0] * 0.6, self.window_size[1]*0.25, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.window_size[0] * 0.35, self.window_size[1]*0.4, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("HIMO Text Descriptions", None)

        if expanded:
            npy_folder = self.label_npy_list[self.label_pid].split('/')[-1]
            imgui.text(str(npy_folder))
            bef_button = imgui.button('<<Before')
            if bef_button:
                self.set_prev_record()
            imgui.same_line()
            next_button = imgui.button('Next>>')
            if next_button:
                self.set_next_record()
            imgui.same_line()
            tmp_idx = ''
            imgui.set_next_item_width(imgui.get_window_width() * 0.1)
            is_go_to, tmp_idx = imgui.input_text('', tmp_idx); imgui.same_line()
            if is_go_to:
                try:
                    self.go_to_idx = int(tmp_idx) - 1
                except:
                    pass
            go_to_button = imgui.button('>>Go<<'); imgui.same_line()
            if go_to_button:
                self.set_goto_record(self.go_to_idx)
            imgui.text(str(self.label_pid+1) + '/' + str(self.total_tasks))

            imgui.text_wrapped(self.text_val)
        imgui.end()

    def set_prev_record(self):
        self.label_pid = (self.label_pid - 1) % self.total_tasks
        self.clear_one_sequence()
        self.load_one_sequence()
        self.scene.current_frame_id=0

    def set_next_record(self):
        self.label_pid = (self.label_pid + 1) % self.total_tasks
        self.clear_one_sequence()
        self.load_one_sequence()
        self.scene.current_frame_id=0

    def set_goto_record(self, idx):
        self.label_pid = int(idx) % self.total_tasks
        self.clear_one_sequence()
        self.load_one_sequence()
        self.scene.current_frame_id=0

    def get_label_file_list(self):
        for clip in sorted(os.listdir(self.clip_folder)):
            if not clip.startswith('.'):
                self.label_npy_list.append(os.path.join(self.clip_folder, clip))
    
    def load_text_from_file(self):
        self.text_val = ''
        clip_name = os.path.split(self.label_npy_list[self.label_pid])[-1][:-4]
        if os.path.exists(os.path.join(self.text_folder, clip_name+'.txt')):
            with open(os.path.join(self.text_folder, clip_name+'.txt'), 'r') as f:
                for line in f.readlines():
                    self.text_val += line
                    self.text_val += '\n'


    def load_one_sequence(self):
        smplx_file = self.label_npy_list[self.label_pid]
        clip_name=os.path.split(smplx_file)[-1][:-4]
        opj_pose_file=os.path.join(self.object_pose_folder, clip_name+'.npy')

        # load smplx

        smplx_params = np.load(smplx_file, allow_pickle=True)
        nf = smplx_params['body_pose'].shape[0]

        betas = smplx_params['betas']
        poses_root = smplx_params['global_orient']
        poses_body = smplx_params['body_pose'].reshape(nf,-1)
        poses_lhand = smplx_params['lhand_pose'].reshape(nf,-1)
        poses_rhand = smplx_params['rhand_pose'].reshape(nf,-1)
        transl = smplx_params['transl']

        # create body models
        smplx_layer = SMPLLayer(model_type='smplx',gender='neutral',num_betas=10,device=C.device)

        # create smplx sequence for two persons
        smplx_seq = SMPLSequence(poses_body=poses_body,
                            smpl_layer=smplx_layer,
                            poses_root=poses_root,
                            betas=betas,
                            trans=transl,
                            poses_left_hand=poses_lhand,
                            poses_right_hand=poses_rhand,
                            device=C.device,
                            )

        self.scene.add(smplx_seq)

        # Load object
        object_pose=np.load(opj_pose_file, allow_pickle=True).item()
        meshes=[]
        for obj_name in object_pose.keys():
            obj_pose=object_pose[obj_name]
            obj_mesh=self.object_mesh[obj_name]
            verts, faces = obj_mesh.vertices, obj_mesh.faces
            mesh = Meshes(
                vertices=verts,
                faces=faces,
                name=obj_name,
                position=obj_pose['transl'],
                rotation=obj_pose['rot'],
                color= (0.3,0.3,0.5,1)
            )
            meshes.append(mesh)
        self.scene.add(*meshes)

        self.load_text_from_file()


    def clear_one_sequence(self):
        for x in self.scene.nodes.copy():
            if type(x) is SMPLSequence or type(x) is SMPLLayer or type(x) is Meshes:
                self.scene.remove(x)


if __name__=='__main__':

    viewer=SMPLX_Viewer()
    viewer.scene.fps=30
    viewer.playback_fps=30
    viewer.run()