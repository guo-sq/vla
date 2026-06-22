import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import optax
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
from openpi.models.pi0_fast import PALIGEMMA_EOS_TOKEN
from openpi.models.pi0_fast import left_to_right_align
from openpi.models.pi0_fast import put_along_last_axis
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b t"],
    embedding_dim: int,
    min_period: float,
    max_period: float,
) -> at.Float[at.Array, "b t {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "bt,d->btd",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05 or config.pi05_subtask_fast
        # subtask + fast discrete action
        self.pi05_subtask_fast = config.pi05_subtask_fast
        self.pi05_with_fast_action = config.pi05_with_fast_action and self.pi05_subtask_fast
        self.fast_action_inference = config.fast_action_inference and self.pi05_with_fast_action
        self.pi05_with_subtask = config.pi05_with_subtask and self.pi05_subtask_fast
        self.subtask_as_action_cond = config.subtask_as_action_cond and self.pi05_with_subtask
        self.infer_action_with_subtask = config.infer_action_with_subtask and self.pi05_with_subtask
        if self.subtask_as_action_cond:
            self.infer_action_with_subtask = True

        self.use_joint_eef_mask = config.use_joint_eef_mask
        self.images_keys = config.image_keys
        self.disable_color_aug = config.disable_color_aug
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                vocab_size=config.vocab_size,
                adarms=self.pi05,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True] if self.pi05 else [False, False],
        )
        # Determine SigLIP head output dimension
        # Use config.vision_output_dim if specified, otherwise use paligemma_config.width
        vision_output_dim = config.vision_output_dim if config.vision_output_dim is not None else paligemma_config.width
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=vision_output_dim,  # Keep internal name for SigLIP compatibility
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if self.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        self.action_out_proj = nnx.Linear(
            action_expert_config.width,
            config.action_dim,
            rngs=rngs,
        )

        self.enable_recap = config.enable_recap
        if self.enable_recap:
            self.optimality_embed_dim = 2048
            self.is_positive_embed = nnx.Embed(2, self.optimality_embed_dim, rngs=rngs)

        if config.enable_rl_value_head:
            # Value head configuration
            self.value_bins = config.value_bins
            self.value_range = config.value_range
            self.value_label_smoothing = config.value_label_smoothing
            self.value_temperature = config.value_temperature

            # Score head components
            # value_bins=1: scalar output with MSE loss (backward compatible)
            # value_bins>1: distributional output with cross-entropy loss
            self.score_out_proj = nnx.Linear(paligemma_config.width, config.value_bins, rngs=rngs)
            self.attn_pool = nnx.Linear(paligemma_config.width, 1, rngs=rngs)
            hidden_dim = max(64, paligemma_config.width // 2)
            self.score_mlp_in = nnx.Linear(paligemma_config.width, hidden_dim, rngs=rngs)
            self.score_mlp_out = nnx.Linear(hidden_dim, paligemma_config.width, rngs=rngs)

            # This attribute gets automatically set by model.train() and model.eval().
            self.deterministic = True
        self.action_loss_weight = config.action_loss_weight
        if self.pi05_subtask_fast:
            if self.pi05_with_subtask:
                self.subtask_loss_weight = config.subtask_loss_weight
            if self.pi05_with_fast_action:
                self.fast_action_loss_weight = config.fast_action_loss_weight
            self.vocab_size = config.vocab_size

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation, *, use_optimality: at.Bool[at.Array, ""] = False
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            # ar_mask += [False] * image_tokens.shape[1]
            ar_mask.append(0 * input_mask[-1])

        tokenized_prompt = obs.tokenized_prompt
        tokenized_prompt_mask = obs.tokenized_prompt_mask
        token_ar_mask = obs.token_ar_mask
        if self.pi05_subtask_fast:
            assert token_ar_mask is not None, "token_ar_mask must be provided for pi0.5 subtask"
            assert tokenized_prompt_mask is not None, "tokenized_prompt_mask must be provided for pi0.5 subtask"
            assert tokenized_prompt is not None, "tokenized_prompt must be provided for pi0.5 subtask"
        # add language (aka tokenized inputs)
        if tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(tokenized_prompt_mask)
            # full attention between image and language inputs
            if not self.pi05_subtask_fast:
                ar_mask.append(0 * tokenized_prompt_mask)
            else:
                ar_mask.append(token_ar_mask)

        if self.enable_recap:
            assert obs.optimality is not None, "Optimality tokens must be provided for RECAP."
            batch_size = tokens[0].shape[0]
            # Conditioned
            con_optimality_inputs = self.is_positive_embed(obs.optimality)
            # Unconditioned
            uncon_optimality_inputs = jnp.zeros(con_optimality_inputs.shape, dtype=tokens[0].dtype)
            # 随机选择使用哪种优度输入
            optimality_inputs = jnp.where(use_optimality, con_optimality_inputs, uncon_optimality_inputs)
            tokens.append(optimality_inputs)

            opt_mask = jnp.broadcast_to(use_optimality, (batch_size, 1))
            input_mask.append(opt_mask)

            ar_mask.append(0 * opt_mask)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        # ar_mask = jnp.array(ar_mask)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b ah"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Int[at.Array, "b s"],
        at.Float[at.Array, "b s emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            # ar_mask += [True]
            ar_mask += [1]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            # time_tokens = einops.repeat(
            #     time_emb, "b emb -> b s emb", s=self.action_horizon
            # )
            time_tokens = time_emb
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        # ar_mask += [True] + ([False] * (self.action_horizon - 1))
        ar_mask += [1] + ([0] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        ar_mask = einops.repeat(
            ar_mask,
            "s -> b s",
            b=tokens.shape[0],
        )
        return tokens, input_mask, ar_mask, adarms_cond

    @at.typecheck
    def score_observation(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "b ..."]:
        """Compute a scalar score for each observation in the batch.

        For distributional value prediction (value_bins > 1), returns logits for each bin.
        For scalar value prediction (value_bins = 1), returns a single scalar per sample.

        This reuses the prefix encoding (images + optional text + optional state)
        and pools the prefix transformer outputs, then applies a linear head
        to produce a single scalar per sample.
        """
        # Preprocess (resizing / masks)
        rng = rng if train else None
        observation = _model.preprocess_observation(rng, observation, train=train)

        # Encode prefix (images + optional language)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        # Forward prefix through LLM (we only need the prefix outputs here).
        (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        return self.compute_score(
            prefix_out,
            prefix_mask,
        )

    def compute_score(
        self,
        prefix_out,
        prefix_mask,
    ):
        # Attention-pooling over tokens: compute per-token logits then softmax over valid tokens
        # prefix_out: [b, s, emb], prefix_mask: [b, s]
        attn_logits = self.attn_pool(prefix_out).squeeze(-1)  # [b, s]
        # Mask out padding tokens: set logits to a large negative value where mask is False
        large_neg = -1e9
        attn_logits = jnp.where(prefix_mask, attn_logits, large_neg)
        attn_weights = jax.nn.softmax(attn_logits, axis=1)[..., None]  # [b, s, 1]
        pooled = jnp.sum(attn_weights * prefix_out, axis=1)  # [b, emb]

        # Small MLP with residual: pooled -> mlp_in -> swish -> mlp_out -> swish -> residual add
        h = self.score_mlp_in(pooled)
        h = nnx.swish(h)
        h = self.score_mlp_out(h)
        h = nnx.swish(h)
        h_out = pooled + h  # residual to preserve prefix signal

        score = self.score_out_proj(h_out)
        # value_bins=1: scalar output in [-1, 1] (backward compatible)
        # value_bins>1: distributional logits output (no sigmoid for cross-entropy)
        if self.value_bins == 1:
            score = jax.nn.sigmoid(score) - 1

        return score

    def compute_rl_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:

        pred_value = self.score_observation(rng, observation, train=train)

        assert observation.returns is not None, "Returns must be provided for RL loss computation."

        # value_bins=1: MSE loss (backward compatible)
        # value_bins>1: Cross-entropy loss for distributional value
        if self.value_bins == 1:
            # pred_value: [b, 1], observation.returns: [b, 1]
            # MSE loss, averaged over batch
            per_sample_loss = jnp.square(pred_value - observation.returns)  # [b, 1]
            value_loss = jnp.mean(per_sample_loss)
        else:
            # Convert returns to bin indices
            # returns in [value_range[0], value_range[1]] -> bin indices in [0, value_bins-1]
            value_min, value_max = self.value_range

            # Normalize returns to [0, 1]
            normalized_returns = (observation.returns - value_min) / (value_max - value_min)

            # Clip to [0, 1] - values outside value_range are clipped to nearest bin
            # This is expected behavior for returns that exceed the configured range
            normalized_returns = jnp.clip(normalized_returns, 0.0, 1.0)

            # Convert to bin indices (int32 is fine here - targets don't need gradients)
            # Using int32 is required by optax.softmax_cross_entropy_with_integer_labels
            target_bins = (normalized_returns * (self.value_bins - 1)).astype(jnp.int32)
            target_bins = jnp.squeeze(target_bins, axis=-1)  # [b]

            # Cross-entropy loss with label smoothing
            # pred_value: [b, value_bins] (logits)
            # target_bins: [b] (int32)
            # Apply label smoothing to reduce over-confidence and smooth predictions
            if self.value_label_smoothing > 0:
                # Manual label smoothing implementation
                # Smoothed target = (1 - smoothing) * one_hot(label) + smoothing * uniform_distribution
                num_classes = self.value_bins
                one_hot_labels = jax.nn.one_hot(target_bins, num_classes)  # [b, num_classes]
                smooth_labels = (
                    1 - self.value_label_smoothing
                ) * one_hot_labels + self.value_label_smoothing / num_classes
                # Use softmax_cross_entropy for soft labels (probabilities)
                per_sample_loss = optax.softmax_cross_entropy(pred_value, smooth_labels)  # [b]
            else:
                per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(pred_value, target_bins)  # [b]
            # Return mean loss over batch
            value_loss = jnp.mean(per_sample_loss)

        return value_loss

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        max_delay: int,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, delay_rng, dropout_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            disable_color_aug=self.disable_color_aug,
            image_keys=self.images_keys,
        )

        # batch_shape = actions.shape[:-2]
        bs, ah, ad = actions.shape
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (bs,)) * 0.999 + 0.001

        # Training-time RTC
        delay = jax.random.randint(delay_rng, (bs,), 0, max_delay)
        action_prefix_mask = jnp.arange(ah)[None, :] < delay[:, None]
        action_postfix_mask = jnp.logical_not(action_prefix_mask)
        time = jnp.where(action_prefix_mask, 0, time[:, None])  # shape: [bs, ah]

        time_expanded = time[..., None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once

        if self.enable_recap:
            use_opt_mask = jax.random.bernoulli(dropout_rng, p=0.7)  # scalar
            # 70% 的概率使用 condition on optimality 的路径
            # 30% 的概率使用 unconditioned 路径
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, use_optimality=use_opt_mask)
        else:
            # 常规路径: 直接计算不带优度引导或固定优度的 v_t
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        # ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        # Build prefix-exclude mask: positions the action expert must NOT attend to.
        # - FAST tokens: always excluded (never action condition).
        # - Subtask tokens: excluded when not subtask_as_action_cond.
        if self.pi05_subtask_fast:
            token_len = observation.tokenized_prompt.shape[1]
            fast_mask = observation.fast_action_loss_mask
            mask_fast = (
                fast_mask[:, :token_len] if fast_mask is not None else jnp.zeros((bs, token_len), dtype=jnp.bool_)
            )
            mask_subtask = (
                (prefix_ar_mask[:, -token_len:].astype(jnp.bool_) & ~mask_fast)
                if not self.subtask_as_action_cond
                else jnp.zeros((bs, token_len), dtype=jnp.bool_)
            )
            prefix_exclude = jnp.concatenate(
                [
                    jnp.zeros((bs, prefix_ar_mask.shape[1] - token_len), dtype=jnp.bool_),
                    mask_fast | mask_subtask,
                ],
                axis=1,
            )
            suffix_exclude = jnp.zeros((bs, suffix_tokens.shape[1]), dtype=jnp.bool_)
            full_exclude = jnp.concatenate([prefix_exclude, suffix_exclude], axis=1)
            start = attn_mask.shape[1] - suffix_tokens.shape[1]
            attn_mask = attn_mask.at[:, start:, :].set(attn_mask[:, start:, :] & ~full_exclude[:, None, :])
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        per_dim_sq_loss = jnp.square(v_t - u_t)
        if not self.use_joint_eef_mask:
            action_loss = jnp.mean(per_dim_sq_loss, axis=-1)
        else:
            joint_eef_dof_mask = observation.joint_eef_dof_mask
            masked_sum = jnp.sum(per_dim_sq_loss * joint_eef_dof_mask, axis=-1)
            mask_count = jnp.sum(joint_eef_dof_mask, axis=-1) + 1e-8
            action_loss = masked_sum / mask_count
        action_loss = action_loss * self.action_loss_weight

        subtask_loss = jnp.zeros(bs)
        fast_action_loss = jnp.zeros(bs)
        if self.pi05_subtask_fast:
            prefix_token_loss_mask = observation.token_loss_mask
            assert prefix_token_loss_mask is not None, "Token loss mask is required"
            prefix_token_prompt = observation.tokenized_prompt
            seq_len = prefix_token_prompt.shape[1]
            targets = jax.nn.one_hot(
                observation.tokenized_prompt[:, 1:],
                self.vocab_size,
            )
            prefix_seq_len = prefix_out.shape[1]
            start_idx = prefix_seq_len - seq_len
            end_idx = prefix_seq_len - 1

            subtask_hidden = prefix_out[:, start_idx:end_idx]
            subtask_logits, _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
                pre_logits=subtask_hidden,
            )
            log_probs = jax.nn.log_softmax(subtask_logits, axis=-1)
            token_losses = -jnp.sum(targets * log_probs, axis=-1)

            loss_mask = prefix_token_loss_mask[:, 1:]
            fast_mask = observation.fast_action_loss_mask
            if fast_mask is not None:
                fast_mask = fast_mask[:, 1:]
                assert fast_mask.shape[1] == loss_mask.shape[1]
            if self.pi05_with_fast_action:
                fast_loss_sum = jnp.sum(token_losses * fast_mask, axis=-1)
                fast_loss_count = jnp.sum(fast_mask, axis=-1)
                fast_action_loss = (
                    jnp.where(
                        fast_loss_count > 0,
                        fast_loss_sum / jnp.maximum(fast_loss_count, 1),
                        0.0,
                    )
                    * self.fast_action_loss_weight
                )
            if self.pi05_with_subtask:
                subtask_mask = loss_mask & ~fast_mask if fast_mask is not None else loss_mask
                subtask_loss_sum = jnp.sum(token_losses * subtask_mask, axis=-1)
                subtask_loss_count = jnp.sum(subtask_mask, axis=-1)
                subtask_loss = (
                    jnp.where(
                        subtask_loss_count > 0,
                        subtask_loss_sum / jnp.maximum(subtask_loss_count, 1),
                        0.0,
                    )
                    * self.subtask_loss_weight
                )

        return (
            action_loss,
            action_postfix_mask,
            per_dim_sq_loss,
            subtask_loss,
            fast_action_loss,
        )

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        action_prefix: _model.Actions,  # shape: [b, ah, ad]
        delay: int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        run_subtask_inference: bool = True,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # π0.5 paper: subtask inference runs at lower frequency than action
        # run_subtask_inference here means whether to run subtask inference at this step.
        ran_subtask = run_subtask_inference and (self.subtask_as_action_cond or self.infer_action_with_subtask)
        generated_subtask_tokens = None
        if ran_subtask:
            rng, subtask_rng = jax.random.split(rng)
            generated_subtask_tokens = self.sample_subtask(
                subtask_rng,
                observation,
                max_decoding_steps=32,
                temperature=0.0,
            )
            observation = self._insert_subtask_tokens_into_observation(observation, generated_subtask_tokens)

        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        # jax.debug.print("num_steps: {n}, dt: {d}", n=num_steps, d=dt)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        if self.enable_recap:
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
                observation, use_optimality=jnp.array(object=True)
            )
        else:
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # rtc_prefix_action_mask.shape: [b, ah]
        rtc_prefix_action_mask = jnp.arange(self.action_horizon)[None, :] < delay

        def step(carry):
            x_t, time = carry
            x_t = jnp.where(rtc_prefix_action_mask[:, :, None], action_prefix, x_t)
            time_cond = jnp.broadcast_to(time, (batch_size, self.action_horizon))
            time_cond = jnp.where(rtc_prefix_action_mask, 0, time_cond)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time_cond)
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            # Exclude from action expert attention:
            # - Subtask tokens: when not subtask_as_action_cond.
            # (No fast_mask: inference outputs continuous actions, no discrete FAST tokens in prefix.)
            if run_subtask_inference and self.infer_action_with_subtask and not self.subtask_as_action_cond:
                # suffix tokens (action tokens) should not attend to subtask tokens in prefix
                # prefix_ar_mask shape: [b, prefix_len], has 1 at subtask positions
                # Need to expand it to match prefix_attn_mask shape: [b, suffix_len, prefix_len]

                # Create subtask mask for prefix: shape [b, prefix_len]
                # 0 at subtask positions, 1 elsewhere (we want to NOT attend to subtask)
                prefix_subtask_mask = ~prefix_ar_mask.astype(jnp.bool_)  # or: prefix_ar_mask == 0

                # Expand to [b, suffix_len, prefix_len] to match prefix_attn_mask
                # prefix_attn_mask was created by repeat: einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
                prefix_subtask_mask_expanded = einops.repeat(
                    prefix_subtask_mask, "b p -> b s p", s=suffix_tokens.shape[1]
                )
                # Apply to the prefix part of full_attn_mask
                prefix_len = prefix_tokens.shape[1]
                full_attn_mask = full_attn_mask.at[:, :, :prefix_len].set(
                    full_attn_mask[:, :, :prefix_len] & prefix_subtask_mask_expanded
                )

            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        actions = jnp.where(rtc_prefix_action_mask[:, :, None], action_prefix, x_0)
        if ran_subtask:
            return {"actions": actions, "subtask_tokens": generated_subtask_tokens}
        return actions

    def value_distribution_to_scalar(
        self,
        value_logits: at.Float[at.Array, "b value_bins"],
        temperature: float | None = None,
    ) -> at.Float[at.Array, "b"]:  # noqa: F821
        """Convert distributional value logits to expected scalar value.

        For distributional value prediction (value_bins > 1), computes the expected value:
            E[V] = Σ_i softmax(logits_i / T) * bin_center_i

        For scalar value prediction (value_bins == 1), returns the input unchanged.

        Temperature scaling (T > 1) makes the softmax distribution smoother, reducing
        temporal fluctuation in value predictions across consecutive frames.

        This implements the inference method described in:
        "π*: Policy Improvement with Online RL for Vision-Language-Action Models"
        https://arxiv.org/pdf/2511.14759

        Args:
            value_logits: Logits for each value bin, shape [batch, value_bins]
            temperature: Optional temperature parameter for softmax scaling. If provided,
                overrides self.value_temperature. This is useful for dynamically adjusting
                the distribution smoothness at inference time without recompilation.

        Returns:
            Expected scalar value, shape [batch]. For value_bins > 1, the value is
            computed as the weighted sum of bin centers using softmax probabilities
            with temperature scaling.
        """
        if self.value_bins == 1:
            return value_logits.squeeze(-1)

        # Use provided temperature parameter, falling back to self.value_temperature
        # Passing temperature as a parameter bypasses JIT caching issues
        temp = temperature if temperature is not None else self.value_temperature

        # Apply temperature scaling before softmax
        # temp > 1 makes distribution smoother, reducing temporal fluctuation
        scaled_logits = value_logits / temp

        # Apply softmax to get probabilities over bins
        probs = jax.nn.softmax(scaled_logits, axis=-1)  # [b, value_bins]

        # Create bin centers uniformly distributed in value_range
        value_min, value_max = self.value_range
        bin_centers = jnp.linspace(value_min, value_max, self.value_bins)  # [value_bins]

        # Compute expected value: E[V] = Σ p_i * v_i
        return jnp.sum(probs * bin_centers, axis=-1)  # [b]

    def _insert_subtask_tokens_into_observation(
        self,
        observation: _model.Observation,
        generated_subtask_tokens: at.Int[at.Array, "b s"],
    ) -> _model.Observation:
        """Insert inferred subtask tokens into observation after valid prompt tokens.

        Used by sample_actions (flow-matching) and _sample_actions_fast when subtask
        is required as action condition. Both share identical insert logic.
        """
        #  找到当前 tokenized_prompt 中有效 token 的数量(排除 padding)
        current_prompt = observation.tokenized_prompt
        current_mask = observation.tokenized_prompt_mask
        current_ar_mask = observation.token_ar_mask
        # 计算每个样本的有效 token 数量
        valid_token_end = jnp.sum(current_mask, axis=1, keepdims=True)
        # 从生成的 subtask tokens 中提取有效的非 padding tokens
        # 假设 sample_subtask 返回的 tokens 中, 0 或特定值表示 padding/未生成
        # 这里需要根据 sample_subtask 的实际返回格式调整
        subtask_valid_mask = generated_subtask_tokens != 0
        # 将 subtask tokens 插入到 valid_token_end 位置
        batch_size, max_len = current_prompt.shape
        max_subtask_len = generated_subtask_tokens.shape[1]
        # 计算实际可以插入的 subtask tokens 数量
        remaining_space = max_len - valid_token_end
        max_insert = jnp.minimum(max_subtask_len, remaining_space)
        # 使用 scatter 操作插入 subtask tokens
        # 创建 indices 用于 scatter
        indices = jnp.arange(max_subtask_len)[None, :]
        insert_positions = valid_token_end + indices
        valid_insert = (indices < max_insert) & subtask_valid_mask

        new_tokenized_prompt = current_prompt.at[jnp.arange(batch_size)[:, None], insert_positions].set(
            jnp.where(
                valid_insert,
                generated_subtask_tokens,
                jnp.take_along_axis(current_prompt, insert_positions, axis=1),
            )
        )
        # 更新 mask
        new_tokenized_prompt_mask = current_mask.at[jnp.arange(batch_size)[:, None], insert_positions].set(valid_insert)
        # 更新 ar_mask
        new_token_ar_mask = current_ar_mask.at[jnp.arange(batch_size)[:, None], insert_positions].set(
            jnp.where(
                valid_insert,
                1,
                jnp.take_along_axis(current_ar_mask, insert_positions, axis=1),
            )
        )
        return observation.replace(
            tokenized_prompt=new_tokenized_prompt,
            tokenized_prompt_mask=new_tokenized_prompt_mask,
            token_ar_mask=new_token_ar_mask,
        )

    def sample_subtask(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int = 32,
        temperature: float = 0.0,
    ) -> jax.Array:
        # # Ensure max_decoding_steps is a concrete Python int to avoid traced value issues
        # if isinstance(max_decoding_steps, jax.core.Tracer):
        #     max_decoding_steps = max_decoding_steps.astype(int)

        # TODO: this is a hack to get the image keys.
        observation = _model.preprocess_observation(None, observation, train=False)

        # embed inputs
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # left to right align all input token sequences
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        [prefix_logits, _], kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            fill_kv_cache=True,
            autoregressive_mode=True,
        )

        # prepare decoding -- final logit decodes the first token
        last_logit = prefix_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

        def step(carry):
            rng, last_logit, output_tokens, cache, _, step = carry
            # last_logit = self.subtask_out_proj(last_logit)
            # Sample token from last logit
            # Split RNG for this step
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping --> stop if all batch elements have EOS token
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            # Decode one step
            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefill_len[:, None] + step + 1
            # Build a full mask over the maximal decoding horizon, then
            # truncate its last axis to match the current kv_cache key length.
            mask_full = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            # `cache` contains the kv_cache returned by the LLM; its second
            # dimension is the current key length. Truncate mask to that length
            # so mask.shape == (batch, 1, kv_key_len) and no mismatch occurs.
            [last_logit, _], kv_cache = self.PaliGemma.llm(
                [token_embedding, None],
                mask=mask_full,
                positions=positions,
                kv_cache=cache,
                fill_kv_cache=True,
                autoregressive_mode=True,
            )
            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # Use lax.while_loop so we can jit the full decoding loop.
        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
        )
        return output_tokens

    def _sample_actions_fast(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        max_decoding_steps: int = 64,
        temperature: float = 0.0,
        generated_subtask_tokens: jax.Array | None = None,
    ) -> dict:
        """Autoregressively decode FAST action tokens. Returns dict with 'tokens' for output transform."""
        if self.pi05_with_subtask:
            if generated_subtask_tokens is None:
                rng, subtask_rng = jax.random.split(rng)
                generated_subtask_tokens = self.sample_subtask(
                    subtask_rng,
                    observation,
                    max_decoding_steps=32,
                    temperature=temperature,
                )
            else:
                generated_subtask_tokens = jnp.asarray(generated_subtask_tokens, dtype=jnp.int32)
            observation = self._insert_subtask_tokens_into_observation(observation, generated_subtask_tokens)

        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        [prefix_logits, _], kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            fill_kv_cache=True,
            autoregressive_mode=True,
        )
        last_logit = prefix_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

        def step(carry):
            rng, last_logit, output_tokens, cache, _, step_idx = carry
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = put_along_last_axis(
                output_tokens,
                jnp.broadcast_to(step_idx, (token.shape[0], 1)),
                token,
            )
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)
            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefill_len[:, None] + step_idx + 1
            mask_full = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < jnp.broadcast_to(
                    prefill_size + step_idx + 1,
                    (prefix_start.shape[0], 1, 1),
                ),
            )
            [last_logit, _], kv_cache = self.PaliGemma.llm(
                [token_embedding, None],
                mask=mask_full,
                positions=positions,
                kv_cache=cache,
                fill_kv_cache=True,
                autoregressive_mode=True,
            )
            return rng, last_logit, output_tokens, kv_cache, all_eos, step_idx + 1

        def cond(carry):
            _, _, _, _, all_eos, step_idx = carry
            return (~all_eos) & (step_idx < max_decoding_steps)

        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
        )

        # ExtractFASTActionsWithSubtask needs full sequence
        # (prefix + decoded) to locate "Action: " in decoded string.
        prefix_tokens = observation.tokenized_prompt
        full_tokens = jnp.concatenate([prefix_tokens, output_tokens], axis=1)
        return {"fast_action_tokens": full_tokens}
