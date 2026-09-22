from __future__ import annotations

import unittest

from policy import TOOL_POLICIES, data_only_boundary, get_tool_policy, tool_metadata


class ToolPoliciesCoverTheAgentToolListsTests(unittest.TestCase):
    """Every tool PydanticAI can actually call must resolve a policy.

    academic_search and github_search had no TOOL_POLICIES entry even though
    they are registered WEB-scope tools: get_tool_policy/tool_metadata would
    raise ValueError the moment anything looked one up for them, exactly the
    way web_search and search_my_files already do via
    safe_metadata["policy"] = tool_metadata(...). This test enumerates the
    real tool lists from agent_runtime._build_agent so a newly added tool
    with no policy entry fails here instead of at runtime.
    """

    def test_every_web_scope_tool_has_a_policy(self) -> None:
        import agent_runtime

        agent = agent_runtime._build_agent(model=None, scope="WEB")
        tool_names = {tool.name for tool in agent._function_toolset.tools.values()}
        # Sanity: confirms the test is actually looking at the real tool set,
        # not an empty/mocked one.
        self.assertIn("web_search", tool_names)
        self.assertIn("academic_search", tool_names)
        self.assertIn("github_search", tool_names)
        for tool_name in tool_names:
            with self.subTest(tool=tool_name):
                get_tool_policy(tool_name)  # must not raise

    def test_every_my_files_scope_tool_has_a_policy(self) -> None:
        import agent_runtime

        agent = agent_runtime._build_agent(model=None, scope="MY_FILES")
        tool_names = {tool.name for tool in agent._function_toolset.tools.values()}
        self.assertIn("search_my_files", tool_names)
        for tool_name in tool_names:
            with self.subTest(tool=tool_name):
                get_tool_policy(tool_name)  # must not raise

    def test_academic_search_and_github_search_metadata_is_well_formed(self) -> None:
        for tool_name in ("academic_search", "github_search"):
            with self.subTest(tool=tool_name):
                metadata = tool_metadata(tool_name)
                self.assertEqual(metadata["tool_name"], tool_name)
                self.assertEqual(metadata["risk_level"], "READ_SAFE")
                self.assertFalse(metadata["requires_approval"])
                self.assertFalse(metadata["external_side_effect"])


class GetToolPolicyTests(unittest.TestCase):
    def test_unknown_tool_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            get_tool_policy("not_a_real_tool")

    def test_every_declared_policy_round_trips_through_tool_metadata(self) -> None:
        for tool_name in TOOL_POLICIES:
            with self.subTest(tool=tool_name):
                metadata = tool_metadata(tool_name)
                self.assertEqual(metadata["tool_name"], tool_name)


class DataOnlyBoundaryTests(unittest.TestCase):
    def test_marks_content_as_untrusted_and_not_instructable(self) -> None:
        boundary = data_only_boundary("github_public_api")
        self.assertEqual(boundary["source"], "github_public_api")
        self.assertEqual(boundary["classification"], "UNTRUSTED_DATA")
        self.assertFalse(boundary["instructions_allowed"])


if __name__ == "__main__":
    unittest.main()
