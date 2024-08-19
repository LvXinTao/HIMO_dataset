import os
import os.path as osp

from loguru import logger
import torch
from src.utils.parser_utils import train_feature_extractor_args
from src.utils.misc import fixseed
from src.utils.word_vectorizer import POS_enumerator,WordVectorizer
from src.feature_extractor.modules import TextEncoderBiGRUCo,MotionEncoderBiGRUCo,MovementConvEncoder
from src.feature_extractor.trainer import TextMotionMatchTrainer
from src.dataset.fe_dataset import feature_extractor_dataset,collate_fn
from src.train.train_platforms import WandbPlatform

from torch.utils.data import DataLoader

def build_models(args):
    movement_enc=MovementConvEncoder(dim_pose,args.dim_movement_enc_hidden,args.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=dim_word,
                                  pos_size=dim_pos_ohot,
                                  hidden_size=args.dim_text_hidden,
                                  output_size=args.dim_coemb_hidden,
                                  device=args.device)
    motion_enc = MotionEncoderBiGRUCo(input_size=args.dim_movement_latent,
                                      hidden_size=args.dim_motion_hidden,
                                      output_size=args.dim_coemb_hidden,
                                      device=args.device)
    
    if not args.is_continue:
       logger.info('Loading Decomp......')
       checkpoint = torch.load(osp.join(args.checkpoints_dir, args.decomp_name, 'model', 'latest.tar'),
                               map_location=args.device)
       movement_enc.load_state_dict(checkpoint['movement_enc'])
    return text_enc,motion_enc,movement_enc

if __name__=='__main__':
    args=train_feature_extractor_args()
    args.device = torch.device("cpu" if args.gpu_id==-1 else "cuda:" + str(args.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    fixseed(args.seed)
    if args.gpu_id!=-1:
        torch.cuda.set_device(args.gpu_id)
    args.save_path=osp.join(args.save_dir,args.exp_name)
    args.model_dir=osp.join(args.save_path,'model')
    args.log_dir=osp.join(args.save_path,'log')
    args.eval_dir=osp.join(args.save_path,'eval')

    os.makedirs(args.save_path,exist_ok=True)
    os.makedirs(args.model_dir,exist_ok=True)
    os.makedirs(args.log_dir,exist_ok=True)
    os.makedirs(args.eval_dir,exist_ok=True)

    logger.add(osp.join(args.log_dir,'train_feature_extractor.log'),rotation='10 MB')

    args.data_root='/data/xuliang/HO2_subsets_original/HO2_final'
    args.max_motion_length=300
    if args.mode=='2o':
        dim_pose=52*6+52*3+3+2*(6+3)
    elif args.mode=='3o':
        dim_pose=52*6+52*3+3+3*(6+3)

    meta_root=osp.join(args.data_root,'glove')
    dim_word=300
    dim_pos_ohot=len(POS_enumerator)

    w_vectorizer=WordVectorizer(meta_root,'himo_vab')
    text_encoder,motion_encoder,movement_encoder=build_models(args)

    pc_text_enc = sum(param.numel() for param in text_encoder.parameters())
    logger.info("Total parameters of text encoder: {}".format(pc_text_enc))
    pc_motion_enc = sum(param.numel() for param in motion_encoder.parameters())
    logger.info("Total parameters of motion encoder: {}".format(pc_motion_enc))
    logger.info("Total parameters: {}".format(pc_motion_enc + pc_text_enc))

    train_platform=WandbPlatform(args.save_path)

    trainer=TextMotionMatchTrainer(args,text_encoder,motion_encoder,movement_encoder,train_platform)

    train_dataset=feature_extractor_dataset(args,'train',w_vectorizer)
    val_dataset=feature_extractor_dataset(args,'val',w_vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, collate_fn=collate_fn, pin_memory=True)
    
    trainer.train(train_loader,val_loader)



    