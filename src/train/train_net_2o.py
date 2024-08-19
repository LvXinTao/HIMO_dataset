import os
import os.path as osp
import json

from src.utils.misc import fixseed,makepath
from src.utils.parser_utils import train_net_2o_args
from src.utils.model_utils import create_model_and_diffusion
from src.utils import dist_utils
from src.train.train_platforms import *
from src.train.training_loop import TrainLoop
from src.dataset.himo_2o_dataset import HIMO_2O
from src.dataset.tensors import himo_2o_collate_fn

from torch.utils.data import DataLoader
from loguru import logger

def main():
    args=train_net_2o_args()
    # save path
    save_path=osp.join(args.save_dir,args.exp_name)
    if osp.exists(save_path):
        raise FileExistsError(f'{save_path} already exists!')
        # pass
    else:
        makepath(save_path)
    args.save_path=save_path
    # training plateform
    train_platform_type=eval(args.train_platform)
    train_platform=train_platform_type(args.save_path)
    train_platform.report_args(args,'Args')
    # config logger
    logger.add(osp.join(save_path,'train.log'))
    # save args
    with open(osp.join(save_path,'args.json'),'w') as f:
        json.dump(vars(args),f,indent=4)

    dist_utils.setup_dist(args.device)
    # get dataset loader
    logger.info('Loading Training dataset...')
    train_dataset=HIMO_2O(args,split='train')
    data_loader=DataLoader(train_dataset,batch_size=args.batch_size,
                           shuffle=True,num_workers=8,drop_last=True,collate_fn=himo_2o_collate_fn)

    # get model and diffusion
    logger.info('Loading Model and Diffusion...')
    model,diffusion=create_model_and_diffusion(args)
    model.to(dist_utils.dev())

    logger.info('Start Training...')
    TrainLoop(args,train_platform,model,diffusion,data_loader).run_loop()
    train_platform.close()

if __name__=='__main__':
    main()