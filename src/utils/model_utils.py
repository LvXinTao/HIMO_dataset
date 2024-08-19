from src.diffusion import gaussian_diffusion as gd
from src.model.net_2o import NET_2O
from src.model.net_3o import NET_3O
from src.diffusion.respace import SpacedDiffusion,space_timesteps

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    
def create_model_and_diffusion(args):
    if args.network=='net_2o':
        model=NET_2O()
        diffusion=create_gaussian_diffusion(args)
    elif args.network=='net_3o':
        model=NET_3O()
        diffusion=create_gaussian_diffusion(args)

    return model,diffusion

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_recon=args.lambda_recon if hasattr(args, 'lambda_recon') else 0.0,
        lambda_pos=args.lambda_pos if hasattr(args, 'lambda_pos') else 0.0,
        lambda_geo=args.lambda_geo if hasattr(args, 'lambda_geo') else 0.0,
        lambda_vel=args.lambda_vel if hasattr(args, 'lambda_vel') else 0.0,
        lambda_sp=args.lambda_sp if hasattr(args, 'lambda_sp') else 0.0,
        train_args=args
    )
