from dextrah_lab.adept.post_training_observer import (
    AdeptPostTrainingObserver,
    _find_task_env,
)


class _Optimizer:
    def __init__(self):
        self.param_groups = [{"lr": 1e-5}]


class _Task:
    def __init__(self):
        self.phases = []

    def set_post_training_phase(self, phase):
        self.phases.append(phase)


class _Wrapper:
    def __init__(self, env):
        self.env = env


class _Algo:
    def __init__(self, task):
        self.vec_env = _Wrapper(_Wrapper(task))
        self.optimizer = _Optimizer()
        self.learning_rate = 1e-5
        self.last_lr = 1e-5
        self.epoch_num = 0


def test_find_task_env_through_wrappers():
    task = _Task()
    assert _find_task_env(_Wrapper(_Wrapper(task))) is task


def test_observer_freezes_actor_then_enables_conservative_lr():
    task = _Task()
    algo = _Algo(task)
    observer = AdeptPostTrainingObserver()
    observer.after_init(algo)
    assert task.phases == ["critic_warmup"]
    assert algo.optimizer.param_groups[0]["lr"] == 0.0

    algo.epoch_num = 19
    observer.after_steps()
    assert task.phases == ["critic_warmup"]

    algo.epoch_num = 20
    observer.after_steps()
    assert task.phases == ["critic_warmup", "ppo"]
    assert algo.optimizer.param_groups[0]["lr"] == 1e-5
