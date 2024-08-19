from argparse import ArgumentParser
import argparse
import os
import json

def parse_and_load_from_model(val_args):
    model_path=val_args.model_path

    # get args from model path args.json
    args_path=os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), f'args.json not found in {os.path.dirname(model_path)}'
    with open(args_path,'r') as f:
        args_json=json.load(f)
    args=argparse.Namespace(**args_json)
    if args.cond_mask_prob==0.0:
        args.guidance_param=1.0
    else:
        args.guidance_param=val_args.guidance_param
    
    return args

def add_base_options(parser):
    group=parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=4396, type=int, help="For fixing random seed.")
    

def add_data_options(parser):
    group=parser.add_argument_group('data')
    group.add_argument("--dataset",default='himo', choices=['himo','eval'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="/data/xuliang/HO2_subsets_original/HO2_final", type=str,
                       help=" dataset directory.")
    
def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")

def add_training_net_2o_options(parser):
    group = parser.add_argument_group('training_net_2o')
    group.add_argument("--save_dir", default='./save', type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--network",type=str,default='net_2o')
    group.add_argument("--exp_name",required=True, type=str,
                       help="Experiment name. Will create a folder with this name in save_dir.")
    group.add_argument("--train_platform",default="WandbPlatform",choices=['WandbPlatform','ClearmlPlatform','TensorboardPlatform','NoPlatform'],type=str,
                       help="Platform for logging training progress.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--num_frames", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--max_motion_length", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--max_text_len",default=40,type=int,help="Max length of text")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--eval_during_training", default=True, type=bool,
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_batch_size", default=32, type=int,)
    group.add_argument("--obj",default='2o',choices=['2o','3o'],type=str,)

    group.add_argument("--lambda_recon", default=1.0, type=float, help="Reconstruction loss weight.")
    group.add_argument("--lambda_pos", default=1.0, type=float, help="Joint Position loss weight.")
    group.add_argument("--lambda_geo",default=1.0,type=float,help="Geometric loss weight.")
    group.add_argument("--lambda_vel", default=1.0, type=float, help="Joint Velocity loss weight.")
    group.add_argument("--lambda_sp",default=1.0,type=float,help="Object Spation Relation loss weight.")

    group.add_argument("--batch_size", default=128, type=int, help="Batch size during training.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.99, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")

    group.add_argument("--log_interval", default=1_00, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=1000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_epochs", default=3000, type=int,
                       help="Number of training epochs.")

def add_training_net_3o_options(parser):
    group = parser.add_argument_group('training_net_3o')
    group.add_argument("--save_dir", default='./save', type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--network",type=str,default='net_3o')
    group.add_argument("--exp_name",required=True, type=str,
                       help="Experiment name. Will create a folder with this name in save_dir.")
    group.add_argument("--train_platform",default="WandbPlatform",choices=['WandbPlatform','ClearmlPlatform','TensorboardPlatform','NoPlatform'],type=str,
                       help="Platform for logging training progress.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--num_frames", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--max_motion_length", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--max_text_len",default=40,type=int,help="Max length of text")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--eval_during_training", default=True, type=bool,
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_batch_size", default=32, type=int,)
    group.add_argument("--obj",default='3o',choices=['2o','3o'],type=str,)

    group.add_argument("--lambda_recon", default=1.0, type=float, help="Reconstruction loss weight.")
    group.add_argument("--lambda_pos", default=1.0, type=float, help="Joint Position loss weight.")
    group.add_argument("--lambda_geo",default=1.0,type=float,help="Geometric loss weight.")
    group.add_argument("--lambda_vel", default=1.0, type=float, help="Joint Velocity loss weight.")
    group.add_argument("--lambda_sp",default=1.0,type=float,help="Object Spation Relation loss weight.")

    group.add_argument("--batch_size", default=128, type=int, help="Batch size during training.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.99, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")

    group.add_argument("--log_interval", default=1_00, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=1000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_epochs", default=3000, type=int,
                       help="Number of training epochs.")

def add_training_mdm_2o_options(parser):
    group = parser.add_argument_group('training_mdm_2o')
    group.add_argument("--save_dir", default='./save', type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--network",type=str,default='mdm_2o')
    group.add_argument("--exp_name",required=True, type=str,
                       help="Experiment name. Will create a folder with this name in save_dir.")
    group.add_argument("--train_platform",default="WandbPlatform",choices=['WandbPlatform','ClearmlPlatform','TensorboardPlatform','NoPlatform'],type=str,
                       help="Platform for logging training progress.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--num_frames", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--max_text_len",default=40,type=int,help="Max length of text")
    group.add_argument("--max_motion_length", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--eval_during_training", default=True, type=bool,
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_batch_size", default=32, type=int,)
    group.add_argument("--obj",default='2o',choices=['2o','3o'],type=str,)

    group.add_argument("--lambda_recon", default=1.0, type=float, help="Reconstruction loss weight.")
    group.add_argument("--lambda_pos", default=1.0, type=float, help="Joint Position loss weight.")
    group.add_argument("--lambda_geo",default=1.0,type=float,help="Geometric loss weight.")
    group.add_argument("--lambda_vel", default=1.0, type=float, help="Joint Velocity loss weight.")

    group.add_argument("--batch_size", default=128, type=int, help="Batch size during training.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.99, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")

    group.add_argument("--log_interval", default=1_00, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=1_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_epochs", default=3000, type=int,
                       help="Number of training epochs.")
    
def add_training_mdm_3o_options(parser):
    group = parser.add_argument_group('training_mdm_3o')
    group.add_argument("--save_dir", default='./save', type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--network",type=str,default='mdm_3o')
    group.add_argument("--exp_name",required=True, type=str,
                       help="Experiment name. Will create a folder with this name in save_dir.")
    group.add_argument("--train_platform",default="WandbPlatform",choices=['WandbPlatform','ClearmlPlatform','TensorboardPlatform','NoPlatform'],type=str,
                       help="Platform for logging training progress.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--num_frames", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--max_motion_length", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--max_text_len",default=40,type=int,help="Max length of text")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--eval_during_training", default=True, type=bool,
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_batch_size", default=32, type=int,)
    group.add_argument("--obj",default='3o',choices=['2o','3o'],type=str,)

    group.add_argument("--lambda_recon", default=1.0, type=float, help="Reconstruction loss weight.")
    group.add_argument("--lambda_pos", default=1.0, type=float, help="Joint Position loss weight.")
    group.add_argument("--lambda_geo",default=1.0,type=float,help="Geometric loss weight.")
    group.add_argument("--lambda_vel", default=1.0, type=float, help="Joint Velocity loss weight.")

    group.add_argument("--batch_size", default=128, type=int, help="Batch size during training.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.99, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")

    group.add_argument("--log_interval", default=1_00, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=1_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_epochs", default=3000, type=int,
                       help="Number of training epochs.")

def add_training_pmdm_2o_options(parser):
    group = parser.add_argument_group('training_pmdm_2o')
    group.add_argument("--save_dir", default='./save', type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--network",type=str,default='pmdm_2o')
    group.add_argument("--exp_name",required=True, type=str,
                       help="Experiment name. Will create a folder with this name in save_dir.")
    group.add_argument("--train_platform",default="WandbPlatform",choices=['WandbPlatform','ClearmlPlatform','TensorboardPlatform','NoPlatform'],type=str,
                       help="Platform for logging training progress.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--num_frames", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--max_motion_length", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--eval_during_training", default=True, type=bool,
                       help="If True, will run evaluation during training.")
    group.add_argument("--obj",default='2o',choices=['2o','3o'],type=str,)

    group.add_argument("--lambda_recon", default=1.0, type=float, help="Reconstruction loss weight.")
    group.add_argument("--lambda_pos", default=1.0, type=float, help="Joint Position loss weight.")
    group.add_argument("--lambda_geo",default=1.0,type=float,help="Geometric loss weight.")
    group.add_argument("--lambda_vel", default=1.0, type=float, help="Joint Velocity loss weight.")

    group.add_argument("--batch_size", default=128, type=int, help="Batch size during training.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.99, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")

    group.add_argument("--log_interval", default=1_00, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=1_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_epochs", default=3000, type=int,
                       help="Number of training epochs.")
    
def add_training_pmdm_3o_options(parser):
    group = parser.add_argument_group('training_pmdm_3o')
    group.add_argument("--save_dir", default='./save', type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--network",type=str,default='pmdm_3o')
    group.add_argument("--exp_name",required=True, type=str,
                       help="Experiment name. Will create a folder with this name in save_dir.")
    group.add_argument("--train_platform",default="WandbPlatform",choices=['WandbPlatform','ClearmlPlatform','TensorboardPlatform','NoPlatform'],type=str,
                       help="Platform for logging training progress.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--num_frames", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--max_motion_length", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--max_motion_length", default=300, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--eval_during_training", default=True, type=bool,
                       help="If True, will run evaluation during training.")
    group.add_argument("--obj",default='3o',choices=['2o','3o'],type=str,)

    group.add_argument("--lambda_recon", default=1.0, type=float, help="Reconstruction loss weight.")
    group.add_argument("--lambda_pos", default=1.0, type=float, help="Joint Position loss weight.")
    group.add_argument("--lambda_geo",default=1.0,type=float,help="Geometric loss weight.")
    group.add_argument("--lambda_vel", default=1.0, type=float, help="Joint Velocity loss weight.")

    group.add_argument("--batch_size", default=128, type=int, help="Batch size during training.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.99, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")

    group.add_argument("--log_interval", default=1_00, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=1_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_epochs", default=3000, type=int,
                       help="Number of training epochs.")
    
def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path",required=True, type=str,
                       help="Path to the model to evaluate.")
    group.add_argument("--eval_mode", required=True, choices=['wo_mm', 'mm_short', 'debug'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results.")
    group.add_argument("--obj",default='2o',choices=['2o','3o'],type=str,
                          help="Number of objects")
    group.add_argument("--batch_size", default=32, type=int,)
    group.add_argument("--num_frames", default=300, type=int,
                       help="Number of samples to generate.")
    group.add_argument("--max_motion_length",type=int,default=300,help="Max length of motion")
    group.add_argument("--max_text_len", type=int, default=40, help="Length of motion")

    group.add_argument('--dim_text_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')
    group.add_argument('--dim_motion_hidden', type=int, default=1024, help='Dimension of hidden unit in GRU')
    group.add_argument('--dim_coemb_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')
    
    group.add_argument("--guidance_param", default=1.0, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.") # 2.5
    

def train_decomp_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default="decomp", help='Name of this trial')
    parser.add_argument("--gpu_id", type=int, default=0,
                                 help='GPU id')
    parser.add_argument("--seed", type=int, default=4396,
                                help='Seed for random')
    parser.add_argument("--mode",type=str,default='2o',choices=['2o','3o'],help='Number of objects')

    parser.add_argument('--checkpoints_dir', type=str, default='./save', help='models are saved here')

    parser.add_argument("--window_size", type=int, default=40, help="Length of motion clips for reconstruction")

    parser.add_argument('--dim_movement_enc_hidden', type=int, default=512,
                                 help='Dimension of hidden in AutoEncoder(encoder)')
    parser.add_argument('--dim_movement_dec_hidden', type=int, default=512,
                                 help='Dimension of hidden in AutoEncoder(decoder)')
    parser.add_argument('--dim_movement_latent', type=int, default=512, help='Dimension of motion snippet')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    parser.add_argument('--max_epoch', type=int, default=270, help='Training iterations')

    parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')

    parser.add_argument('--lambda_sparsity', type=float, default=0.001, help='Layers of GRU')
    parser.add_argument('--lambda_smooth', type=float, default=0.001, help='Layers of GRU')

    parser.add_argument('--lr', type=float, default=1e-4, help='Layers of GRU')

    parser.add_argument('--is_continue', action="store_true", help='Training iterations')

    parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
    parser.add_argument('--save_every_e', type =int, default=10, help='Frequency of printing training progress')
    parser.add_argument('--eval_every_e', type=int, default=3, help='Frequency of printing training progress')
    parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')
    return parser.parse_args()

def train_feature_extractor_args():
    parser=ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="fe_epoch300", help='Name of this trial')
    parser.add_argument("--gpu_id", type=int, default=0,
                                help='GPU id')
    parser.add_argument("--mode",type=str,default='2o',choices=['2o','3o'],help='Number of objects')
    parser.add_argument("--seed", type=int, default=4396,
                                help='Seed for random')
    parser.add_argument("--is_train", action="store_true",default=True,
                         help='Training iterations')
    
    parser.add_argument("--dataset", type=str, default="feature_extractor")
    parser.add_argument('--save_dir', type=str, default='./save', help='models are saved here')
    parser.add_argument('--decomp_name', type=str, default="decomp", help='Name of this trial')
    parser.add_argument('--checkpoints_dir', type=str, default='./save', help='models are saved here')

    parser.add_argument("--unit_length", type=int, default=5, help="Length of motion")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    parser.add_argument("--max_motion_length",type=int,default=300,help="Max length of motion")
    parser.add_argument("--max_text_len", type=int, default=40, help="Length of motion")


    parser.add_argument('--dim_movement_enc_hidden', type=int, default=512,
                                 help='Dimension of hidden unit in GRU')
    parser.add_argument('--dim_movement_latent', type=int, default=512, help='Dimension of hidden unit in GRU')
    
    parser.add_argument('--dim_text_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')
    parser.add_argument('--dim_motion_hidden', type=int, default=1024, help='Dimension of hidden unit in GRU')
    parser.add_argument('--dim_coemb_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')

    parser.add_argument('--max_epoch', type=int, default=300, help='Training iterations')
    parser.add_argument('--estimator_mod', type=str, default='bigru')
    parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')
    parser.add_argument('--negative_margin', type=float, default=10.0)

    parser.add_argument('--lr', type=float, default=1e-4, help='Layers of GRU')

    parser.add_argument('--is_continue', action="store_true", help='Training iterations')

    parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
    parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of printing training progress')
    parser.add_argument('--eval_every_e', type=int, default=5, help='Frequency of printing training progress')
    parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')

    return parser.parse_args()

def train_net_2o_args():
    parser=ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_diffusion_options(parser)
    add_training_net_2o_options(parser)
    return parser.parse_args()

def train_net_3o_args():
    parser=ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_diffusion_options(parser)
    add_training_net_3o_options(parser)
    return parser.parse_args()

def train_mdm_2o_args():
    parser=ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_diffusion_options(parser)
    add_training_mdm_2o_options(parser)
    return parser.parse_args()

def train_mdm_3o_args():
    parser=ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_diffusion_options(parser)
    add_training_mdm_3o_options(parser)
    return parser.parse_args()

def train_pmdm_2o_args():
    parser=ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_diffusion_options(parser)
    add_training_pmdm_2o_options(parser)
    return parser.parse_args()

def train_mdm_3o_args():
    parser=ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_diffusion_options(parser)
    add_training_pmdm_3o_options(parser)
    return parser.parse_args()

def eval_himo_args():
    parser=ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_evaluation_options(parser)
    val_args=parser.parse_args()
    model_args=parse_and_load_from_model(val_args)
    return val_args,model_args
