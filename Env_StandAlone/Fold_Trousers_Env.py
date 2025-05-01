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
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from omni.physx.scripts import deformableUtils,particleUtils,physicsUtils

# load custom package
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment
from Env_Config.Garment.Deformable_Garment import Deformable_Garment
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Position_Judge import judge_pcd
from Env_Config.Utils_Project.Point_Cloud_Manip import rotate_point_cloud
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation

class FoldTrousers_Env(BaseEnv):
    def __init__(
        self, 
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
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
            usd_path="Assets/Garment/Trousers/Long/PL_Pants036/PL_Pants036_obj.usd" if usd_path is None else usd_path,
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
            dexleft_pos=np.array([-0.8, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )
        
        # load camera
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, 1.0, 6.75]), 
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([0.0, 4.0, 6.0]),
            camera_orientation=np.array([0, 60, -90.0]),
            prim_path="/World/env_camera",
        )
        
        self.garment_pcd = None
        self.points_affordance_feature = None
        
        # load GAM Model
        self.model = GAM_Encapsulation(catogory="Trousers")        
        
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # initialize world
        self.reset()
        
        # move garment to the target position
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)
        self.position = [pos[0], pos[1], 0.2]
        self.orientation = ori
        
        # initialize recording camera to obtain point cloud data of garment
        self.garment_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment",
            ]
        )
        # initialize gif camera to obtain rgb with the aim of creating gif
        self.env_camera.initialize(depth_enable=True)
        
        # add thread and record gif Asynchronously(use to collect rgb data for generating gif)
        if record_vedio_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_vedio)
            self.thread_record.daemon = True
        
        # open hand to be initial state
        self.bimanual_dex.set_both_hand_state("open", "open")

        # step world to make it ready
        for i in range(100):
            self.step()

        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint(f"usd_path: {usd_path}", "magenta")
        cprint(f"pos_x: {pos[0]}", "magenta")
        cprint(f"pos_y: {pos[1]}", "magenta")
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
                "points_affordance_feature": self.points_affordance_feature,
            })
        
        self.step_num += 1
    
def FoldTrousers(pos, ori, usd_path, ground_material_usd, data_collection_flag, record_vedio_flag):
        
    env = FoldTrousers_Env(pos, ori, usd_path, ground_material_usd, record_vedio_flag)
   
    if record_vedio_flag:
        env.thread_record.start()     # if you want to record gif, please uncomment this line
    
    # hide prim to get garment point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
    for i in range(50):
        env.step()
        
    pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )
    env.garment_pcd=pcd
    
    # unhide
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=True,
    )
    for i in range(50):
        env.step()
    
    pcd_rotate = rotate_point_cloud(pcd, euler_angles=np.array([0, 0, -90]), center_point=env.garment.get_garment_center_pos())     
    # get manipulation points from GAM Model
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd_rotate, index_list=[983, 521, 1801, 471, 1170, 214])

    manipulation_points = pcd[indices]
    # print("actual_manipulation_points:\n", manipulation_points)
    
    manipulation_points[:, 2] = 0.0
    
    # ------------------------------- fold procedure 1 ----------------------------------------------
        
    env.points_affordance_feature = normalize_columns(points_similarity[0:2].T)
        
    env.bimanual_dex.dense_move_both_ik(
        left_pos=manipulation_points[0], 
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]), 
        right_pos=manipulation_points[1], 
        right_ori=np.array([0.406, -0.406, -0.579, 0.579])
    )
    
    if data_collection_flag:
        for i in range(20):
            env.step()
        env.record(task_name="Fold_Trousers", stage_index=1)
    
    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")
            
    height=abs(manipulation_points[3][1]-manipulation_points[1][1])/2
    # lift
    lift_points_1 = np.array([manipulation_points[0][0]-0.02, manipulation_points[0][1]+height/2, height+0.02])
    lift_points_2 = np.array([manipulation_points[1][0]+0.02, manipulation_points[1][1]+height/2, height+0.02])
    
    env.bimanual_dex.dense_move_both_ik(
        left_pos=lift_points_1,
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        right_pos=lift_points_2,
        right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
    )
    
    # move to destination
    lift_points_1 = np.array([manipulation_points[2][0]-0.03, manipulation_points[2][1], height+0.03])
    lift_points_2 = np.array([manipulation_points[3][0]+0.03, manipulation_points[3][1], height+0.03])
    
    env.bimanual_dex.dense_move_both_ik(
        left_pos=lift_points_1, 
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]), 
        right_pos=lift_points_2, 
        right_ori=np.array([0.406, -0.406, -0.579, 0.579])
    )
    
    # drop
    lift_points_1 = np.array([manipulation_points[2][0]-0.02, manipulation_points[2][1]+0.01, 0.08])
    lift_points_2 = np.array([manipulation_points[3][0]+0.03, manipulation_points[3][1]+0.01, 0.08])
    
    env.bimanual_dex.dense_move_both_ik(
        left_pos=lift_points_1,
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        right_pos=lift_points_2,
        right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
    )
    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
    
    if data_collection_flag:
        env.stop_record()
    
    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(100):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0)
    
    # move_away
    lift_points_1 = np.array([manipulation_points[2][0], manipulation_points[2][1], 0.2])
    lift_points_2 = np.array([manipulation_points[3][0], manipulation_points[3][1], 0.2])
    
    env.bimanual_dex.dense_move_both_ik(
        left_pos=lift_points_1, 
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]), 
        right_pos=lift_points_2, 
        right_ori=np.array([0.406, -0.406, -0.579, 0.579])
    )
    
    # move away
    lift_points_1 = np.array([-0.6, 0.8, 0.5])
    lift_points_2 = np.array([0.6, 0.8, 0.5])
    
    env.bimanual_dex.dense_move_both_ik(
        left_pos=lift_points_1, 
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]), 
        right_pos=lift_points_2, 
        right_ori=np.array([0.406, -0.406, -0.579, 0.579])
    )
    
    for i in range(100):
        env.step()
    
    
    # ------------------------------- fold procedure 2 ----------------------------------------------
        
    env.points_affordance_feature = normalize_columns(np.concatenate([points_similarity[4:5], points_similarity[4:5]], axis=0).T)
        
    # move to catch point
    # env.bimanual_dex.dexright.dense_step_action(
    #     target_pos=np.array([manipulation_points[4][0], manipulation_points[4][1], 0.5]),
    #     target_ori=np.array([0.0, 0.0, 0.707, -0.707]),  
    #     angular_type="quat",                                                                                 
    # )
    
    # catch garment
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=manipulation_points[4],
        target_ori=np.array([0.0, 0.0, 0.707, -0.707]),  
        angular_type="quat",                                                                                 
    )
    
    if data_collection_flag:
        for i in range(20):
            env.step()
        env.record(task_name="Fold_Trousers", stage_index=2)
    
    env.bimanual_dex.set_both_hand_state(left_hand_state="None", right_hand_state="close")
    
    length=abs(manipulation_points[4][0]-manipulation_points[5][0])/2
    # lift
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=np.array([manipulation_points[4][0]-length, manipulation_points[4][1], length+0.02]),
        target_ori=np.array([0.0, 0.0, 0.707, -0.707]),   
        angular_type="quat",                                                                                
    )
    
    # move to destination
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=np.array([manipulation_points[5][0]-0.0, manipulation_points[5][1], 0.2]),
        target_ori=np.array([0.0, 0.0, 0.707, -0.707]),   
        angular_type="quat",                                                                                
    )
    env.bimanual_dex.set_both_hand_state(left_hand_state="None", right_hand_state="open")
    
    if data_collection_flag:
        env.stop_record()
    
    # release garment
    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(100):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0)
    
    dexleft_prim = prims_utils.get_prim_at_path("/World/DexLeft")
    dexright_prim = prims_utils.get_prim_at_path("/World/DexRight")
    set_prim_visibility(dexleft_prim, False)
    set_prim_visibility(dexright_prim, False)
    
    for i in range(50):
        env.step()   
        
    success=True
    _,indices,x=env.model.get_manipulation_points(pcd_rotate,[216,471,1288])
    points=pcd[indices]
    boundary=[points[0][0]-0.1,(points[0][0]+points[1][0])/2+0.05,points[2][1]-0.1,points[0][1]+0.1]
    pcd_end,_=env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )
    success=judge_pcd(pcd_end,boundary,threshold=0.15)
    cprint(f"final result: {success}", color="green", on_color="on_green")
    
    # if you wanna create gif, use this code. Need Cooperation with thread.
    if record_vedio_flag and success:
        if not os.path.exists("Data/Fold_Trousers/vedio"):
            os.makedirs("Data/Fold_Trousers/vedio")
        env.env_camera.create_mp4(get_unique_filename("Data/Fold_Trousers/vedio/vedio", ".mp4"))

    if data_collection_flag:
        # write into .log file
        with open("Data/Fold_Trousers/data_collection_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}\n")
        if success:
            env.record_to_npz()
            if not os.path.exists("Data/Fold_Trousers/final_state_pic"):
                os.makedirs("Data/Fold_Trousers/final_state_pic")
            env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Fold_Trousers/final_state_pic/img",".png"))
    
    
if __name__=="__main__":
    
    args=parse_args_record()
    
    # initial setting
    pos = np.array([0.0, 0.7, 0.2])
    ori = np.array([0.0, 0.0, 90.0])
    usd_path = None
    
    if args.garment_random_flag:
        np.random.seed(int(time.time()))
        x = np.random.uniform(-0.1, 0.1) # changeable
        y = np.random.uniform(0.7, 0.8) # changeable
        pos = np.array([x,y,0.0])
        ori = np.array([0.0, 0.0, 90.0])
        Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_lists = os.path.join(Base_dir,"Model_HALO/GAM/checkpoints/Trousers/assets_list.txt")
        assets_list = []
        with open(assets_lists,"r",encoding='utf-8') as f:
            for line in f:
                clean_line = line.rstrip('\n')
                assets_list.append(clean_line)
        usd_path=np.random.choice(assets_list)
    
    FoldTrousers(pos, ori, usd_path, args.ground_material_usd, args.data_collection_flag, args.record_vedio_flag)

    if args.data_collection_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()
    
simulation_app.close()
