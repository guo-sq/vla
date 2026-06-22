"""Tests for TaskSpec / RoleSpec prompt template substitution."""

import pytest

from lerobot.recording.task.task_spec import (
    RoleSpec,
    TaskSpec,
    _substitute_template,
)


class TestSubstituteTemplate:
    def test_simple_dotted_path(self):
        out = _substitute_template(
            "Pack the {socks.color} socks.",
            {"socks": {"color": "white"}},
        )
        assert out == "Pack the white socks."

    def test_top_level_key(self):
        out = _substitute_template("Hello {operator}.", {"operator": "alice"})
        assert out == "Hello alice."

    def test_unresolvable_left_alone_by_default(self):
        # {speed} is the canonical per-episode placeholder; the session-time
        # pass must not consume it.
        out = _substitute_template(
            "Do it under {speed} seconds.",
            {"socks": {"color": "white"}},
        )
        assert out == "Do it under {speed} seconds."

    def test_strict_mode_raises_on_unknown(self):
        with pytest.raises(KeyError):
            _substitute_template("Hello {missing}.", {}, strict=True)

    def test_partial_resolution_in_mixed_template(self):
        out = _substitute_template(
            "Pack the {socks.color} socks under {speed} seconds.",
            {"socks": {"color": "blue"}},
        )
        assert out == "Pack the blue socks under {speed} seconds."

    def test_non_dict_traversal_left_alone(self):
        # If the path tries to traverse a non-dict, leave the placeholder
        # untouched in lenient mode.
        out = _substitute_template(
            "Value is {socks.color.deeper}.",
            {"socks": {"color": "white"}},
        )
        assert out == "Value is {socks.color.deeper}."

    def test_value_coerced_to_str(self):
        out = _substitute_template(
            "Robot id {id}.", {"id": 4}
        )
        assert out == "Robot id 4."

    def test_empty_template_unchanged(self):
        assert _substitute_template("", {"x": "y"}) == ""

    def test_no_placeholders_unchanged(self):
        assert _substitute_template("Plain text.", {"x": "y"}) == "Plain text."


class TestRoleSpecApplyTemplate:
    def test_substitutes_prompt(self):
        role = RoleSpec(prompt="Pack the {socks.color} socks.", max_time_s=60)
        out = role.apply_template({"socks": {"color": "white"}})
        assert out.prompt == "Pack the white socks."
        # original is not mutated
        assert role.prompt == "Pack the {socks.color} socks."

    def test_speed_placeholder_survives(self):
        role = RoleSpec(prompt="Hang under {speed}s.", max_time_s=60)
        out = role.apply_template({"socks": {"color": "white"}})
        assert out.prompt == "Hang under {speed}s."

    def test_combined_session_and_per_episode(self):
        role = RoleSpec(
            prompt="Pack the {socks.color} socks under {speed} seconds.",
            max_time_s=60,
        )
        # Session pass leaves {speed} for get_prompt_for_episode.
        substituted = role.apply_template({"socks": {"color": "blue"}})
        assert substituted.prompt == "Pack the blue socks under {speed} seconds."


class TestTaskSpecApplyTemplate:
    def _spec(self) -> TaskSpec:
        return TaskSpec(
            task_id="pack_socks",
            roles={
                "operator": RoleSpec(prompt="Pack the {socks.color} socks.", max_time_s=60),
            },
        )

    def test_substitutes_all_role_prompts(self):
        spec = self._spec()
        out = spec.apply_template({"socks": {"color": "blue"}})
        assert out.roles["operator"].prompt == "Pack the blue socks."

    def test_returns_new_instance_unchanged_original(self):
        spec = self._spec()
        spec.apply_template({"socks": {"color": "blue"}})
        assert spec.roles["operator"].prompt == "Pack the {socks.color} socks."

    def test_multiple_roles_each_substituted(self):
        spec = TaskSpec(
            task_id="multi",
            roles={
                "builder": RoleSpec(prompt="Build the {socks.color} fortress.", max_time_s=60),
                "destroyer": RoleSpec(prompt="Destroy the {socks.color} fortress.", max_time_s=45),
            },
        )
        out = spec.apply_template({"socks": {"color": "red"}})
        assert out.roles["builder"].prompt == "Build the red fortress."
        assert out.roles["destroyer"].prompt == "Destroy the red fortress."
