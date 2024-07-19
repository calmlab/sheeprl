# MsPacman  https://gymnasium.farama.org/environments/atari/ms_pacman/
# sheeprl exp=dreamer_v3 env=atari env.id=MsPacmanDeterministic-v4
# sheeprl exp=dreamer_v3 env=atari env.id=MsPacmanNoFrameskip-v4
sheeprl exp=dreamer_v3_100k_ms_pacman fabric.accelerator=cuda env.num_envs=4