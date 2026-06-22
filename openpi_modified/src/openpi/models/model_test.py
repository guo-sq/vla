from flax import nnx
import jax
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True)
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    nnx_utils.module_jit(model.compute_loss)(key, obs, act, 0)
    # assert loss.shape == (batch_size, config.action_horizon)
    action_prefix = jax.random.normal(key, (batch_size, config.action_horizon, config.action_dim))
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, action_prefix=action_prefix, delay=0, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act, 0)
    if isinstance(loss, tuple):
        loss = loss[0]
    assert loss.shape == (batch_size, config.action_horizon)

    action_prefix = jax.random.normal(key, (batch_size, config.action_horizon, config.action_dim))
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, action_prefix=action_prefix, delay=0, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act, 0)
    if isinstance(loss, tuple):
        loss = loss[0]
    assert loss.shape == (batch_size, config.action_horizon)

    action_prefix = jax.random.normal(key, (batch_size, config.action_horizon, config.action_dim))
    actions = model.sample_actions(key, obs, action_prefix=action_prefix, delay=0, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_subtask_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05_subtask_fast=True,
        pi05_with_subtask=True,
        pi05_with_fast_action=True,
        max_token_len=128,
        subtask_as_action_cond=True,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    nnx_utils.module_jit(model.compute_loss)(key, obs, act, 0)
    # assert loss.shape == (batch_size, config.action_horizon)
    action_prefix = jax.random.normal(key, (batch_size, config.action_horizon, config.action_dim))
    sample_fn = nnx_utils.module_jit(model.sample_actions, static_argnames=("run_subtask_inference",))
    results = sample_fn(
        key,
        obs,
        action_prefix=action_prefix,
        delay=0,
        num_steps=10,
        run_subtask_inference=True,
    )
    actions = results["actions"]
    subtask_tokens = results["subtask_tokens"]
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
    assert subtask_tokens.shape == (batch_size, 32)
    sample_actions_fast = getattr(model, "_sample_actions_fast", None)
    assert sample_actions_fast is not None
    discrete_actions = nnx_utils.module_jit(sample_actions_fast)(key, obs)
    discrete_actions = discrete_actions["fast_action_tokens"]
    # full_tokens = prefix (max_token_len) + decoded (max_decoding_steps)
    assert discrete_actions.shape[0] == batch_size
    assert discrete_actions.shape[1] >= config.max_token_len  # prefix + decoded
    print(subtask_tokens.shape, discrete_actions.shape, actions.shape)


if __name__ == "__main__":
    test_pi0_subtask_model()
