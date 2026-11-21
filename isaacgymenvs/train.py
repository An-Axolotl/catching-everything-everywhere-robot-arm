# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import datetime
import isaacgym

import os
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import to_absolute_path
import gym
#import ml_runlog
import sys
import os
import rl_games
import inspect
import yaml
from gym import logger
logger.set_level(logger.INFO)

sys.path.append('..')
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

from isaacgymenvs.utils.utils import set_np_formatting, set_seed #, get_current_commit_hash

## OmegaConf & Hydra Config

def verify_package_locations():
    ig_path = os.path.relpath(inspect.getfile(isaacgym))

    if ig_path != "../isaacgym/python/isaacgym/__init__.py":
        raise Exception("isaacgym package location is not as expected: {}".format(ig_path))
    
    rl_games_path = os.path.relpath(inspect.getfile(rl_games))

    if rl_games_path != "../rl_games/rl_games/__init__.py":
        raise Exception("rl_games package location is not as expected: {}".format(rl_games_path))


# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    #verify_package_locations()

    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs
    import shutil

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.default_run_name}_{time_str}" if 'run_name' not in cfg else cfg.run_name
    run_url = None

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        #cfg.checkpoint = to_absolute_path(cfg.checkpoint)
        cfg.checkpoint = to_absolute_path(f'runs/{cfg.run_name}/nn/{cfg.checkpoint}')


    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device = f'cuda:{rank}'

    # sets seed. if seed is -1 will pick a random one
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)

    if cfg.wandb_activate and rank == 0:
        # Make sure to install WandB if you actually use this.

        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            resume="allow",
            name=cfg.run_name if 'run_name' in cfg else None
        )


        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.run_name = wandb.run.name
        # if wandb is activated use wandb assigned run name
        run_name = cfg.run_name
        run_url = wandb.run.url
    
    if cfg.test and cfg.checkpoint:
        print('REPLACED TRAIN CONFIG WITH CHECKPOINT CONFIG')
        cfg.task = OmegaConf.load(f'runs/{cfg.run_name}/{cfg.task.name}.yaml')

    with open_dict(cfg):
        cfg.task['env']['checkpoint'] = str(cfg.checkpoint) if cfg.checkpoint else None
        cfg.task['env']['test'] = bool(cfg.test) if cfg.test else False
        cfg.task['env']['real'] = bool(cfg.real) if 'real' in cfg else False
    
    cfg.train.params.config.full_experiment_name = run_name # This name is used to save checkpoints

    experiment_dir = os.path.join('runs', run_name)

    def create_env_thunk(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )

        # if not cfg.capture_video and cfg.record_intermediate_models:
        #     from record import record_episode
        #     from functools import partial
        #     from rl_games.common.a2c_common import A2CBase
        #     A2CBase.record_episode = partial(record_episode, task=cfg.task.name, run_name=run_name, resolution=cfg.capture_resolution, capture_length=cfg.capture_video_len)
            
        if cfg.capture_video:
            envs.render_mode = "rgb_array"
            envs.metadata['render_modes'] = ['rgb_array']
            envs.is_vector_env = True
            #os.makedirs(f'{experiment_dir}/videos', exist_ok=True)

            def trigger_record(step):
                if step >= cfg.capture_video_len + 100: 
                    exit()
                return step == 0
                
            envs = gym.wrappers.RecordVideo(
                envs,
                f"{experiment_dir}/video-{cfg.video_epoch}",
                step_trigger=trigger_record,
                video_length=cfg.capture_video_len,
            )

        if not cfg.headless:
            from pynput.keyboard import Controller
            keyboard = Controller()  # Create the controller
            keyboard.type('\t')
            keyboard.type(']')
            keyboard.type(']')

        return envs


    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': create_env_thunk,
    })

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    # convert CLI arguments into dictionory
    rlg_config_dict = omegaconf_to_dict(cfg.train)
    # import numpy as np
    # from gym import spaces
    # rlg_config_dict["params"]["config"]["env_info"] = {}
    # num_obs = 110
    # num_actions = 22
    # observation_space = spaces.Box(np.ones(num_obs) * -np.Inf, np.ones(num_obs) * np.Inf)
    # rlg_config_dict["params"]["config"]["env_info"]["observation_space"] = observation_space
    # action_space = spaces.Box(np.ones(num_actions) * -1., np.ones(num_actions) * 1.)
    # rlg_config_dict["params"]["config"]["env_info"]["action_space"] = action_space
    # rlg_config_dict["params"]["config"]["env_info"]["agents"] = 1


    # create runner and set the settings
    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()


    # dump config dict
    os.makedirs(experiment_dir, exist_ok=True)
    if not cfg.test:
        shutil.copyfile("cfg/task/Overhand.yaml", os.path.join(experiment_dir, "Overhand.yaml"))
        shutil.copyfile("cfg/train/OverhandPPO.yaml", os.path.join(experiment_dir, "OverhandPPO.yaml"))
        shutil.copyfile("tasks/overhand.py", os.path.join(experiment_dir, "overhand.py"))
        shutil.copyfile("tasks/sdf.py", os.path.join(experiment_dir, "sdf.py"))
        shutil.copyfile("train.py", os.path.join(experiment_dir, "train.py"))

        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    if cfg.wandb_activate and rank == 0:
        wandb.save(os.path.join(experiment_dir, 'config.yaml'))
        wandb.save(os.path.join(experiment_dir, 'OverhandPPO.yaml'))
        wandb.save(os.path.join(experiment_dir, 'Overhand.yaml'))
        wandb.save(os.path.join(experiment_dir, 'overhand.py'))
        wandb.save(os.path.join(experiment_dir, 'sdf.py'))
        wandb.save(os.path.join(experiment_dir, 'train.py'))
        
    if cfg.log_to_sheet:
        ml_runlog.init(
            os.path.expanduser(cfg.creds_path),
            cfg.sheet_name
        )

        ml_runlog.log_data(
            datetime=time_str,
            run_name=run_name,
            run_url=run_url,
            machine=os.uname().nodename,
            log_dir=experiment_dir,
            commit_hash=get_current_commit_hash(),
            cmdline_args=";".join(sys.argv[1:]),
            comments=os.getenv("SHEET_LOGGER")
        )

    if cfg.multi_gpu:
        import horovod.torch as hvd
        hvd.init()
        rank = hvd.rank()
        print('gpu rank', rank)
    else:
        rank = 0

    os.system("rm -rf ~/.isaacgym/vhacd")

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint' : cfg.checkpoint,
        'sigma' : None
    })

    #print("Catch Activated")

    #wandb.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})

    # if cfg.record_final_model and rank == 0:
    #     print("Starting recording")
    #     from record import record_episode
    #     print("Import")
    #     print(f"{experiment_dir}/nn/{cfg.task.name}.pth")
    #     if os.path.exists(f"{experiment_dir}/nn/{cfg.task.name}.pth"):
    #         record_episode(cfg.task.name, epoch='best', task=cfg.task.name, run_name=run_name, resolution=cfg.capture_resolution, capture_length=cfg.capture_video_len)
    #         video_dir = f'runs/{run_name}/video-best'
    #         wandb.log({"video/best":wandb.Video(f'{video_dir}/rl-video-step-0.mp4')})

    print("Done")
    if cfg.wandb_activate and rank == 0:
        wandb.finish()

if __name__ == "__main__":
    launch_rlg_hydra()
