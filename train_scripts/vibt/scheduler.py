from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput
import torch


class ViBTScheduler(UniPCMultistepScheduler):
    def __init__(self, **kwargs):
        super().__init__(**{**kwargs, "use_flow_sigmas": True})
        self.set_parameters()

    def set_parameters(self, noise_scale=1.0, shift_gamma=5.0, seed=None):
        self.noise_scale = noise_scale
        self.config.flow_shift = shift_gamma
        self.generator = (
            None
            if seed is None
            else torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        )

    def step(self, model_output, timestep, sample, return_dict: bool = True, **kwargs):
        delta_t = (
            torch.max(self.timesteps[self.timesteps < timestep]) - timestep
            if (self.timesteps < timestep).any()
            else -timestep - 1
        ) / 1000

        current_t = (timestep + 1) / 1000.0
        eta = (-delta_t * (current_t + delta_t) / current_t) ** 0.5

        noise = torch.randn(
            sample.shape,
            generator=self.generator,
            device=sample.device,
            dtype=sample.dtype,
        )
        latents = sample + delta_t * model_output + eta * self.noise_scale * noise

        if not return_dict:
            return (latents,)
        return SchedulerOutput(prev_sample=latents)

    @classmethod
    def from_scheduler(
        cls, scheduler: UniPCMultistepScheduler, noise_scale=1.0, shift_gamma=5.0
    ):
        obj = cls.__new__(cls)
        # 复制底层 UniPC 调度器的运行时状态
        obj.__dict__ = scheduler.__dict__.copy()

        # 确保 UniPCMultistepScheduler 期望的属性存在（避免 __getattr__ 走 config）
        if not hasattr(obj, "solver_p"):
            obj.solver_p = None

        # 初始化 ViBT 自身的参数（噪声尺度 / flow_shift / 随机数）
        obj.set_parameters(noise_scale, shift_gamma)
        return obj