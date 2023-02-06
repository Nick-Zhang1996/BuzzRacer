from gym.envs.registration import register
register(
    id='BuzzRacer-v0',
    entry_point='BuzzRacer:BuzzRacerEnv',
    max_episode_steps=1000,
)
