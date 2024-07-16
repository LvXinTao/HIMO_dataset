import os
import time

import glfw
import imgui
import numpy as np
import trimesh

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
from aitviewer.renderables.skeletons import Skeletons

glfw.init()
primary_monitor = glfw.get_primary_monitor()
mode = glfw.get_video_mode(primary_monitor)
width = mode.size.width
height = mode.size.height

C.update_conf({'window_width': width*0.9, 'window_height': height*0.9})

OPTITRACK_LIMBS=[
[0,1],[1,2],[2,3],[3,4],
[0,5],[5,6],[6,7],[7,8],
[0,9],[9,10],
[10,11],[11,12],[12,13],[13,14],
    [14,15],[15,16],[16,17],[17,18],
    [14,19],[19,20],[20,21],[21,22],
    [14,23],[23,24],[24,25],[25,26],
    [14,27],[27,28],[28,29],[29,30],
    [14,31],[31,32],[32,33],[33,34],
[10,35],[35,36],[36,37],[37,38],
    [38,39],[39,40],[40,41],[41,42],
    [38,43],[43,44],[44,45],[45,46],
    [38,47],[47,48],[48,49],[49,50],
    [38,51],[51,52],[52,53],[53,54],
    [38,55],[55,56],[56,57],[57,58],
[10,59],[59,60]
]

SELECTED_JOINTS=np.concatenate(
[range(0,5),range(6,10),range(11,63)]
)

class Skel_Viewer(Viewer):
    title='HIMO Viewer for Skeleton' 

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

        self.clip_folder = os.path.join('data','joints')
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
        skel_file = self.label_npy_list[self.label_pid]
        clip_name=os.path.split(skel_file)[-1][:-4]
        opj_pose_file=os.path.join(self.object_pose_folder, clip_name+'.npy')

        # load skeleton
        skel_data = np.load(skel_file, allow_pickle=True)
        skel_data=skel_data[:,SELECTED_JOINTS]
        skel=Skeletons(
            joint_positions=skel_data,
            joint_connections=OPTITRACK_LIMBS,
            radius=0.005
        )
        self.scene.add(skel)

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
            if type(x) is Skeletons or type(x) is Meshes:
                self.scene.remove(x)


if __name__=='__main__':

    viewer=Skel_Viewer()
    viewer.scene.fps=30
    viewer.playback_fps=30
    viewer.run()