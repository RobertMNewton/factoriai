from .environment import FactorioEnv
from . import HumanFeedbackReward, AbstractReward, FactorioFileReward

from src.bot.bot import Bot
from src.model.model import Model
from src.config import Config, default_config

from torch import optim

import tianshou as ts


def rl_train_loop(model: Model, reward_fn: AbstractReward, config: Config = default_config, epsiode_time: int = 300, lr: float = 6E-5, replay_buffer_size: int = 1000):
    env = FactorioEnv(
        Bot(
            model,
            config.keys,
            config.mouse_space,
            config.window_space,
        ),
        reward_fn,
        config.keys,
        config.delays,
        config.mouse_space,
        episode_time=epsiode_time,
    )
    
    policy = ts.policy.DQNPolicy(
        model,
        optim.Adam(model.parameters(), lr=lr),
        target_update_freq=epsiode_time
    )
    
    collector = ts.data.Collector(
        policy,
        env,
        ts.data.ReplayBuffer(replay_buffer_size),
        exploration_noise=True,
    )
    
    result = ts.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=collector,
        test_collector=None,
        max_epoch=10, step_per_epoch=300, step_per_collect=4*30,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
    ).run()
    
    print(f'Finished training! Use {result}')

