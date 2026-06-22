import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def score_observation(self, obs: Dict) -> Dict:  # noqa: UP006
        """Compute value score for the observation.

        This is an optional feature for models with RL value head.
        Default implementation raises NotImplementedError.

        Returns:
            dict with:
                - "value": scalar value (float)
                - "value_logits": raw logits (np.ndarray)
                - "value_metadata": dict with model config
                - "policy_timing": timing info
        """
        raise NotImplementedError("score_observation not supported by this policy")

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
