from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# load external package
import os
import sys
import time
import numpy as np
import open3d as o3d
from termcolor import cprint
import threading

# load isaac-relevant package
import omni.replicator.core as rep
import isaacsim.core.utils.prims as prims_utils
from pxr import UsdGeom,UsdPhysics,PhysxSchema, Gf
from isaacsim.core.api import World
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility, delete_prim
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from isaacsim.core.utils.rotations import euler_angles_to_quat
from omni.physx.scripts import deformableUtils,particleUtils,physicsUtils

# load custom package
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment
from Env_Config.Garment.Deformable_Garment import Deformable_Garment
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Room.Object_Tools import hanger_load, set_prim_visible_group, delete_prim_group
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Point_Cloud_Manip import rotate_point_cloud
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation

class HangTrousers_Env(BaseEnv):
    def __init__(
        self, 
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
        env_dx:float=0.0,
        env_dy:float=0.0,
        ground_material_usd:str=None,
        record_vedio_flag:bool=False, 
    ):
        # load BaseEnv
        super().__init__()

        
        # ------------------------------------ #
        # ---        Add Env Assets        --- #
        # ------------------------------------ #
        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = ground_material_usd,
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )
        
        # load garment
        self.garment = Particle_Garment(
            self.world, 
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            usd_path="Assets/Garment/Trousers/Long/PL_074/PL_074_obj.usd" if usd_path is None else usd_path,
            # contact_offset=0.012,             
            # rest_offset=0.010,                
            # particle_contact_offset=0.012,    
            # fluid_rest_offset=0.010,
            # solid_rest_offset=0.010,
        )
        # Here are some example garments you can try:
        # "Assets/Garment/Trousers/Long/PL_Pants051/PL_Pants051_obj.usd",
        # "Assets/Garment/Trousers/Long/PL_085/PL_085_obj.usd",
        # "Assets/Garment/Trousers/Long/PL_Pants091/PL_Pants091_obj.usd",
        # "Assets/Garment/Trousers/Long/PL_LongPants024/PL_LongPants024_obj.usd",
        # "Assets/Garment/Trousers/Long/PL_M1_074/PL_M1_074_obj.usd",
        # "Assets/Garment/Trousers/Long/PL_M2_015/PL_M2_015_obj.usd",            
        # "Assets/Garment/Trousers/Long/PL_041/PL_041_obj.usd",

        # load bimanual_dex
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.8, 0.0, 0.6]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.0, 0.6]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )
        
        # load camera
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, -3.0, 6.75]), 
            camera_orientation=np.array([0, 60.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([0.0, 6.65, 4.0]),
            camera_orientation=np.array([0, 30.0, -90.0]),
            prim_path="/World/env_camera",
        )
        
        self.object_camera = Recording_Camera(
            camera_position=np.array([0.0, -6.6, 4.9]),
            camera_orientation=np.array([0, 30.0, 90.0]),
            prim_path="/World/object_camera",
        )

        self.garment_pcd = None
        self.object_pcd = None
        self.points_affordance_feature = None
        
        # load GAM Model
        self.model = GAM_Encapsulation(catogory="Trousers")        
        
        # load hanger
        self.env_dx = env_dx
        self.env_dy = env_dy
        self.hanger_center = hanger_load(self.scene, env_dx, env_dy)
        
        self.judge_camera = Recording_Camera(
            camera_position=np.array([self.hanger_center[0], 6.0, 0.5]),
            camera_orientation=np.array([0, 0.0, -90.0]),
            prim_path="/World/judge_camera",
        )
        
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # initialize world
        self.reset()

        # initialize recording camera to obtain point cloud data of garment
        self.garment_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment",
            ]
        )
        # initialize gif camera to obtain rgb with the aim of creating gif
        self.env_camera.initialize(
            depth_enable=True,
        )
        
        self.judge_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment",
            ]
        )
        
        self.object_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/hanger1",
                "/World/hanger2",
                "/World/hanger3",
            ]
        )
        
        # add thread and record gif Asynchronously(use to collect rgb data for generating gif)
        if record_vedio_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_vedio)
            self.thread_record.daemon = True
            
        # move garment to the target position
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)
        self.position = [pos[0], pos[1], 0.2]
        self.orientation = ori
                
        # open hand to be initial state
        self.bimanual_dex.set_both_hand_state("open", "open")

        # step world to make it ready
        for i in range(200):
            self.step()
            
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint(f"usd_path: {usd_path}", "magenta")
        cprint(f"pos_x: {pos[0]}", "magenta")
        cprint(f"pos_y: {pos[1]}", "magenta")
        cprint(f"env_dx: {env_dx}", "magenta")
        cprint(f"env_dy: {env_dy}", "magenta")
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])


        cprint("World Ready!", "green", "on_green")


    def record_callback(self, step_size):

        if self.step_num % 5 == 0:
        
            joint_pos_L = self.bimanual_dex.dexleft.get_joint_positions()
            
            joint_pos_R = self.bimanual_dex.dexright.get_joint_positions()
            
            joint_state = np.array([*joint_pos_L, *joint_pos_R])

            rgb = self.env_camera.get_rgb_graph(save_or_not=False)

            point_cloud = self.env_camera.get_pointcloud_from_depth(
                show_original_pc_online=False,
                show_downsample_pc_online=False,
            )
            
            self.saving_data.append({ 
                "joint_state": joint_state,
                "image": rgb,
                "env_point_cloud": point_cloud,
                "garment_point_cloud":self.garment_pcd,
                "object_point_cloud":self.object_pcd,
                "points_affordance_feature": self.points_affordance_feature,
            })
        
        self.step_num += 1

        
# if __name__=="__main__":
def HangTrousers(pos, ori, usd_path, env_dx, env_dy, ground_material_usd, data_collection_flag, record_vedio_flag):
    
    env = HangTrousers_Env(pos, ori, usd_path, env_dx, env_dy, ground_material_usd, record_vedio_flag)
    
    env.garment.particle_material.set_gravity_scale(0.7)
    
    # hide prim to get object point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Garment/garment"],
        visible=False,
    )
    for i in range(50):
        env.step()
    
    env.object_pcd, color = env.object_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        # real_time_watch=True,
    )
    
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Garment/garment"],
        visible=True,
    )
    for i in range(50):
        env.step()

    # hide prim to get garment point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/hanger1", "/World/hanger2", "/World/hanger3"],
        visible=False,
    )
    for i in range(50):
        env.step()
            
    env.garment_pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        # real_time_watch=True,
    )

    # make prim visible
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/hanger1", "/World/hanger2", "/World/hanger3"],
        visible=True,
    )
    for i in range(50):
        env.step()
        
    if record_vedio_flag:
        env.thread_record.start()

    # get manipulation points from GAM Model
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=env.garment_pcd, index_list=[1083,219])
    
    if abs(manipulation_points[0][0]-manipulation_points[1][0])<0.05:
        print("model false")
        simulation_app.close()
        
    env.points_affordance_feature = normalize_columns(points_similarity.T)
        
    # get lift height
    y_min = env.garment_pcd[np.argmin(env.garment_pcd[:, 1])]
    y_max = env.garment_pcd[np.argmax(env.garment_pcd[:, 1])]
    garment_length=y_max[1]-y_min[1]
    lift_height=env.hanger_center[2]+garment_length/2

    cprint(f"lift height: {lift_height}", "blue")
    
    manipulation_points[:, 2] = 0.00  # set z-axis to 0.025 to make sure dexhand can grasp the garment

    # move both dexhand to the manipulation points
    env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0], left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=manipulation_points[1], right_ori=np.array([0.406, -0.406, -0.579, 0.579]))
    
    if data_collection_flag:
        for i in range(20):
            env.step()
        env.record(task_name="Hang_Trousers", stage_index=1)
    
    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")

    dis=np.sqrt((manipulation_points[0][0]-manipulation_points[1][0])**2+(manipulation_points[0][1]-manipulation_points[1][1])**2)/2

    left_lift_points, right_lift_points = np.array([-dis+env.hanger_center[0], env.hanger_center[1]-0.25,0.5]), np.array([dis+env.hanger_center[0], env.hanger_center[1]-0.25, 0.5])    # set z-axis to 0.65 to make sure dexhand can lift the garment
    
    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    left_lift_points, right_lift_points = np.array([-dis+env.hanger_center[0],env.hanger_center[1]-0.25,0.9]), np.array([dis+env.hanger_center[0],env.hanger_center[1]-0.25, 0.9])    # set z-axis to 0.65 to make sure dexhand can lift the garment

    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    
    if lift_height >1.3:
        lift_height=1.3
        
    left_lift_points, right_lift_points = np.array([-dis+env.hanger_center[0]-0.02, env.hanger_center[1]+0.10, lift_height]), np.array([dis+env.hanger_center[0]+0.02, env.hanger_center[1]+0.10, lift_height])  
    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))
    

    left_lift_points, right_lift_points = np.array([-dis+env.hanger_center[0]-0.02,env.hanger_center[1]+garment_length/2.5, 0.9]), np.array([dis+env.hanger_center[0]+0.02,env.hanger_center[1]+garment_length/2.5, 0.9])  
    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
    
    if data_collection_flag:
        env.stop_record()
        
    env.garment.particle_material.set_gravity_scale(2.0)
    
    for i in range(200):
        env.step()
    
    env.garment.particle_material.set_gravity_scale(0.7)
    
    # make prim visible
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
    for i in range(50):
        env.step()
  
    success=True
    
    pcd_judge, _ = env.judge_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        # real_time_watch=True
    )
    z_values = pcd_judge[:, 2]  # 假设 pcd_judge 的形状是 (N, 3)
    points_below_threshold = np.sum(z_values < 0.01)
    if points_below_threshold > 15:
        success=False
    elif env.garment.get_garment_center_pos()[0] > env.hanger_center[0]+0.05 or env.garment.get_garment_center_pos()[0] < env.hanger_center[0]-0.05:
        success=False
    else:
        success=True

    cprint("----------- Judge Begin -----------", "blue", attrs=["bold"])
    cprint(f"points_below_threshold: {points_below_threshold}", "blue")
    cprint(f"garment_center_pos_x: {env.garment.get_garment_center_pos()[0]}", "blue")
    cprint(f"hanger_center_x: {env.hanger_center[0]}", "blue")
    cprint("----------- Judge End -----------", "blue", attrs=["bold"])
    cprint(f"final result: {success}", color="green", on_color="on_green")
    
    # if you wanna create gif, use this code. Need Cooperation with thread.
    if record_vedio_flag and success:
        if not os.path.exists("Data/Hang_Trousers/vedio"):
            os.makedirs("Data/Hang_Trousers/vedio")
        env.env_camera.create_mp4(get_unique_filename("Data/Hang_Trousers/vedio/vedio", ".mp4"))

    if data_collection_flag:
        # write into .log file
        with open("Data/Hang_Trousers/data_collection_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}  env_dx:{env_dx}  env_dy:{env_dy} \n")

    if data_collection_flag:
        if success:
            env.record_to_npz(env_change=True)
            if not os.path.exists("Data/Hang_Trousers/final_state_pic"):
                os.makedirs("Data/Hang_Trousers/final_state_pic")
            env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Hang_Trousers/final_state_pic/img",".png"))


if __name__=="__main__":
    
    args = parse_args_record()
    
    # initial setting
    pos = np.array([0, 0.7, 0.2])
    ori = np.array([0.0, 0.0, 0.0])
    usd_path = None
    env_dx = 0.0
    env_dy = 0.0 

    if args.env_random_flag or args.garment_random_flag:
        np.random.seed(int(time.time()))
        if args.env_random_flag:
            env_dx = np.random.uniform(-0.15, 0.15) # changeable
            env_dy = np.random.uniform(-0.05, 0.05) # changeable
        if args.garment_random_flag:
            x = np.random.uniform(-0.1, 0.1) # changeable
            y = np.random.uniform(0.5, 0.8) # changeable
            pos = np.array([x,y,0.0])
            ori = np.array([0.0, 0.0, 0.0])
            Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            assets_lists = os.path.join(Base_dir,"Model_HALO/GAM/checkpoints/Trousers/assets_training_list.txt")
            assets_list = []
            with open(assets_lists,"r",encoding='utf-8') as f:
                for line in f:
                    clean_line = line.rstrip('\n')
                    assets_list.append(clean_line)
            usd_path=np.random.choice(assets_list)

    HangTrousers(pos, ori, usd_path, env_dx, env_dy, args.ground_material_usd, args.data_collection_flag, args.record_vedio_flag)

    if args.data_collection_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()
    
simulation_app.close()