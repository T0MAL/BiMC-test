from yacs.config import CfgNode as CN
from utils.util import set_gpu, set_seed
from utils.phase1_fusion import (
    VALID_CNN_EXPERIMENT_MODES,
    VALID_CNN_PROJECTIONS,
    VALID_FUSION_BETA_MODES,
    VALID_FUSION_GEOMETRIES,
    VALID_RELIABILITY_MODES,
)
import argparse

def print_args(cfg):
    print("************")
    print("** Config **")
    print("************")
    print(cfg)
    print("************")


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """

    # Device setting
    cfg.DEVICE = CN()
    cfg.DEVICE.DEVICE_NAME = ''
    cfg.DEVICE.GPU_ID = ''

    cfg.METHOD = ''
    cfg.SEED = -1
    cfg.OUTPUT_DIR = 'results/phase1'
    cfg.RUN_NAME = ''
    cfg.PHASE1_TIMESTAMP = ''

    # For dataset config
    cfg.DATASET = CN()
    cfg.DATASET.NAME = ''
    cfg.DATASET.ROOT = ''
    cfg.DATASET.GPT_PATH = ''
    cfg.DATASET.NUM_CLASSES   = -1
    cfg.DATASET.NUM_INIT_CLS  = -1
    cfg.DATASET.NUM_INC_CLS   = -1
    cfg.DATASET.NUM_BASE_SHOT = -1
    cfg.DATASET.NUM_INC_SHOT  = -1
    cfg.DATASET.BETA = -1.0
    cfg.DATASET.ENSEMBLE_ALPHA = -1.0
    
    # For data
    cfg.DATALOADER = CN()
    cfg.DATALOADER.TRAIN = CN()
    cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE = -1
    cfg.DATALOADER.TRAIN.BATCH_SIZE_INC = -1
    cfg.DATALOADER.TEST = CN()
    cfg.DATALOADER.TEST.BATCH_SIZE = -1
    cfg.DATALOADER.NUM_WORKERS = -1

    # For model
    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ''

    # For methods
    cfg.TRAINER = CN()
    cfg.TRAINER.BiMC = CN()
    cfg.TRAINER.BiMC.PREC = ''
    cfg.TRAINER.BiMC.VISION_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_I = -1.0
    cfg.TRAINER.BiMC.TAU = -1
    cfg.TRAINER.BiMC.TEXT_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_T = -1.0
    cfg.TRAINER.BiMC.GAMMA_BASE = -1.0
    cfg.TRAINER.BiMC.GAMMA_INC = -1.0
    cfg.TRAINER.BiMC.USING_ENSEMBLE = False
    cfg.TRAINER.BiMC.FUSION_BETA_MODE = 'fixed'
    cfg.TRAINER.BiMC.FUSION_GEOMETRY = 'linear'
    cfg.TRAINER.BiMC.BETA_TEMPERATURE = 0.05
    cfg.TRAINER.BiMC.BETA_CLIP_MIN = 0.05
    cfg.TRAINER.BiMC.BETA_CLIP_MAX = 0.95
    cfg.TRAINER.BiMC.RELIABILITY_MODE = 'entropy_margin'
    cfg.TRAINER.BiMC.SAVE_PHASE1_REPORT = True
    cfg.TRAINER.BiMC.USE_CNN_BRANCH = False
    cfg.TRAINER.BiMC.CNN_BACKBONE = 'resnet50'
    cfg.TRAINER.BiMC.CNN_EXPERIMENT_MODE = 'none'
    cfg.TRAINER.BiMC.CNN_LAMBDA = 0.10
    cfg.TRAINER.BiMC.CNN_TOPK = 5
    cfg.TRAINER.BiMC.CNN_PROJECTION = 'random_orthogonal'
    cfg.TRAINER.BiMC.CNN_CACHE_FEATURES = True



    

def setup_cfg(dataset_cfg_file, method_cfg_file):
    cfg = CN()
    extend_cfg(cfg)

    # 1. From the dataset config file
    cfg.merge_from_file(dataset_cfg_file)

    # 2. From the method config file
    cfg.merge_from_file(method_cfg_file)

    cfg.freeze()
    return cfg


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')


def apply_cli_overrides(cfg, args):
    cfg.defrost()
    if args.fusion_beta_mode is not None:
        cfg.TRAINER.BiMC.FUSION_BETA_MODE = args.fusion_beta_mode
    if args.fusion_geometry is not None:
        cfg.TRAINER.BiMC.FUSION_GEOMETRY = args.fusion_geometry
    if args.beta_temperature is not None:
        cfg.TRAINER.BiMC.BETA_TEMPERATURE = args.beta_temperature
    if args.beta_clip_min is not None:
        cfg.TRAINER.BiMC.BETA_CLIP_MIN = args.beta_clip_min
    if args.beta_clip_max is not None:
        cfg.TRAINER.BiMC.BETA_CLIP_MAX = args.beta_clip_max
    if args.reliability_mode is not None:
        cfg.TRAINER.BiMC.RELIABILITY_MODE = args.reliability_mode
    if args.save_phase1_report is not None:
        cfg.TRAINER.BiMC.SAVE_PHASE1_REPORT = args.save_phase1_report
    if args.use_cnn_branch is not None:
        cfg.TRAINER.BiMC.USE_CNN_BRANCH = args.use_cnn_branch
    if args.cnn_backbone is not None:
        cfg.TRAINER.BiMC.CNN_BACKBONE = args.cnn_backbone
    if args.cnn_experiment_mode is not None:
        cfg.TRAINER.BiMC.CNN_EXPERIMENT_MODE = args.cnn_experiment_mode
    if args.cnn_lambda is not None:
        cfg.TRAINER.BiMC.CNN_LAMBDA = args.cnn_lambda
    if args.cnn_topk is not None:
        cfg.TRAINER.BiMC.CNN_TOPK = args.cnn_topk
    if args.cnn_projection is not None:
        cfg.TRAINER.BiMC.CNN_PROJECTION = args.cnn_projection
    if args.cnn_cache_features is not None:
        cfg.TRAINER.BiMC.CNN_CACHE_FEATURES = args.cnn_cache_features
    if args.output_dir is not None:
        cfg.OUTPUT_DIR = args.output_dir
    if args.run_name is not None:
        cfg.RUN_NAME = args.run_name
    if args.phase1_timestamp is not None:
        cfg.PHASE1_TIMESTAMP = args.phase1_timestamp
    if args.seed is not None:
        cfg.SEED = args.seed
    cfg.freeze()
    return cfg


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run the pipeline")

    parser.add_argument('--data_cfg', type=str, help="Path to the data configuration file")
    parser.add_argument('--train_cfg', type=str, help="Path to the training configuration file")
    parser.add_argument('--fusion_beta_mode', type=str, choices=VALID_FUSION_BETA_MODES)
    parser.add_argument('--fusion_geometry', type=str, choices=VALID_FUSION_GEOMETRIES)
    parser.add_argument('--beta_temperature', type=float)
    parser.add_argument('--beta_clip_min', type=float)
    parser.add_argument('--beta_clip_max', type=float)
    parser.add_argument('--reliability_mode', type=str, choices=VALID_RELIABILITY_MODES)
    parser.add_argument('--save_phase1_report', type=str2bool, nargs='?', const=True)
    parser.add_argument('--use_cnn_branch', type=str2bool, nargs='?', const=True)
    parser.add_argument('--cnn_backbone', type=str)
    parser.add_argument('--cnn_experiment_mode', type=str, choices=VALID_CNN_EXPERIMENT_MODES)
    parser.add_argument('--cnn_lambda', type=float)
    parser.add_argument('--cnn_topk', type=int)
    parser.add_argument('--cnn_projection', type=str, choices=VALID_CNN_PROJECTIONS)
    parser.add_argument('--cnn_cache_features', type=str2bool, nargs='?', const=True)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--phase1_timestamp', type=str)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    data_cfg = args.data_cfg
    train_cfg = args.train_cfg

    cfg = setup_cfg(data_cfg, train_cfg)
    cfg = apply_cli_overrides(cfg, args)

    # Set the random seed and GPU ID
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    # Import and run the trainer
    from engine.engine import Runner
    engine = Runner(cfg)
    engine.run()


if __name__ == '__main__':
    main()
