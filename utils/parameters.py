import os
import argparse
import omegaconf
from loguru import logger
def get_parameters(config="config.yaml"):
    # add cmd args
    # priority: cmd args > dict in config
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config', type=str, default=config)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--sync-bn', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    # parse the above cmd options
    args_cmd = parser.parse_known_args()[0]    
    args_cmd.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args_cmd.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args_cmd.debug:
        args_cmd.train.epochs = 2
    args_cmd_dict = vars(args_cmd)

    #add config dict
    try:
        oc_cfg = omegaconf.OmegaConf.load(args_cmd.config)
    except Exception as e:
        logger.warning(f'config path [{args_cmd.config}] is not valid.')
        raise e
    oc_cfg.merge_with(args_cmd_dict)  # oc_cfg is dict: cmd + configs
    
    return oc_cfg


if __name__ == "__main__":
    config = get_parameters("config/config.yaml")
    print(config)