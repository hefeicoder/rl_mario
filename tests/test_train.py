from src.train import build_model


def test_build_ppo_model_smoke():
    model, env = build_model(algo="ppo", n_envs=1, tb_log=None)
    model.learn(total_timesteps=64)  # tiny smoke run
    env.close()


def test_build_dqn_model_smoke():
    model, env = build_model(algo="dqn", n_envs=1, tb_log=None)
    model.learn(total_timesteps=64)
    env.close()
