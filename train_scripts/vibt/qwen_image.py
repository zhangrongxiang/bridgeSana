from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image


from diffusers import QwenImageEditPipeline, QwenImagePipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import retrieve_latents


def encode_vae_image(pipe, image: torch.Tensor, generator: torch.Generator):
    latent_channels = pipe.vae.config.z_dim if getattr(pipe, "vae", None) else 16
    image_latents = retrieve_latents(
        pipe.vae.encode(image), generator=generator, sample_mode="argmax"
    )
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, latent_channels, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    latents_std = (
        torch.tensor(pipe.vae.config.latents_std)
        .view(1, latent_channels, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    image_latents = (image_latents - latents_mean) / latents_std

    return image_latents


@torch.no_grad()
def encode_image(pipe: QwenImagePipeline, image):
    width, height = image.size
    image = pipe.image_processor.preprocess(image, height, width)
    image = image.to(dtype=pipe.dtype, device=pipe.device).unsqueeze(2)
    image_latents = encode_vae_image(pipe, image, None)

    image_latent_height, image_latent_width = image_latents.shape[3:]
    image_latents = pipe._pack_latents(
        image_latents,
        1,
        pipe.transformer.config.in_channels // 4,
        image_latent_height,
        image_latent_width,
    )
    return image_latents


@torch.no_grad()
def decode_latents_image(pipe: QwenImagePipeline, latents):
    latents = pipe._unpack_latents(latents, 1024, 1024, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image


aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}


def process_input_img(image):
    # find the closest aspect ratio
    w, h = image.size
    aspect_ratio = w / h
    closest_ratio = min(
        aspect_ratios.items(),
        key=lambda x: abs((x[1][0] / x[1][1]) - aspect_ratio),
    )
    target_size = closest_ratio[1]
    return image.resize(target_size, Image.LANCZOS)


@torch.no_grad()
def qwen_bridge_gen(
    self: QwenImageEditPipeline,
    image: Optional[PipelineImageInput] = None,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    true_cfg_scale: float = 4.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    # Bridge specific
    return_trajectory=False,
):
    image_size = image[0].size if isinstance(image, list) else image.size
    calculated_width, calculated_height = image_size
    height = height or calculated_height
    width = width or calculated_width

    multiple_of = self.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # 3. Preprocess image
    if image is not None and not (
        isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels
    ):
        image = self.image_processor.resize(image, calculated_height, calculated_width)
        prompt_image = image
        image = self.image_processor.preprocess(
            image, calculated_height, calculated_width
        )
        image = image.unsqueeze(2)

    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    prompt_embeds, prompt_embeds_mask = self.encode_prompt(
        image=prompt_image,
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
            image=prompt_image,
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    noise, image_latents = self.prepare_latents(
        image,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    latents = image_latents.clone()
    img_shapes = [
        [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]
    ] * batch_size

    # 5. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    self._num_timesteps = len(timesteps)

    # handle guidance
    guidance = None
    txt_seq_lens = (
        prompt_embeds_mask.sum(dim=1).tolist()
        if prompt_embeds_mask is not None
        else None
    )
    negative_txt_seq_lens = (
        negative_prompt_embeds_mask.sum(dim=1).tolist()
        if negative_prompt_embeds_mask is not None
        else None
    )

    trajectory = [latents] if return_trajectory else None

    # 6. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t

            latent_model_input = latents

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    # img_shapes=[[(1, 64, 64)]],
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs={},
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                with self.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        encoder_hidden_states=negative_prompt_embeds,
                        img_shapes=img_shapes,
                        # img_shapes=[[(1, 64, 64)]],
                        txt_seq_lens=negative_txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                comb_pred = neg_noise_pred + true_cfg_scale * (
                    noise_pred - neg_noise_pred
                )

                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

            # # step
            # next_t = timesteps[i + 1] if i < len(timesteps) - 1 else 0

            # sigma_t = t / 1000
            # sigma_next_t = next_t / 1000
            # sigma_delta = sigma_next_t - sigma_t
            # print(
            #     f"sigma_t: {sigma_t}, sigma_next_t: {sigma_next_t}, sigma_delta: {sigma_delta}"
            # )

            # noise = torch.randn(
            #     latents.shape,
            #     dtype=latents.dtype,
            #     device=latents.device,
            #     generator=generator,
            # )
            # eta = torch.sqrt(-sigma_delta * sigma_next_t / sigma_t)
            # # eta = torch.sqrt(-sigma_delta)

            # coef = torch.clip(noise_pred.abs(), 0, 1) if rescale_noise else 1
            # latents = latents + noise_pred * sigma_delta + sigma * eta * noise * coef
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if return_trajectory:
                trajectory.append(latents)

            # call the callback, if provided
            progress_bar.update()

    self._current_timestep = None
    if output_type == "latent":
        image = latents
    else:

        def decode_latents(latents, height, width):
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            return image

        image = decode_latents(latents, height, width)
        trajectory = (
            [decode_latents(t, height, width)[0] for t in trajectory]
            if return_trajectory
            else None
        )

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    if return_trajectory:
        return QwenImagePipelineOutput(images=image), trajectory
    else:
        return QwenImagePipelineOutput(images=image)