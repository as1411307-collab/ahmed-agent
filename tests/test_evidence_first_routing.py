from __future__ import annotations

import unittest

from request_routing import RequestCapability, classify_request


class EvidenceFirstRoutingTests(unittest.TestCase):
    def test_operational_evidence_cases_do_not_fall_back_to_general(self) -> None:
        cases = (
            (
                "استخدم المنظومة كاملة، استخدم المنظومة الموحدة.",
                RequestCapability.PROJECT_STATE,
            ),
            (
                "قيّم كل المسارات ولا تمشي على رأيي لمجرد أني قلته، اختار الأفضل.",
                RequestCapability.PROJECT_STATE,
            ),
            (
                "أي كتابة أو إجراء خارجي يحتاج موافقتي الصريحة قبل التنفيذ.",
                RequestCapability.PROJECT_STATE,
            ),
            (
                "راقب تحديثات OpenAI وChatGPT الرسمية ونبهني فقط إذا تغير شيء يؤثر فعليًا على قرار build-vs-buy أو إعداد الوكيل.",
                RequestCapability.EXTERNAL_WEB_RESEARCH,
            ),
            (
                "توكن المالك ما ينحط في الرابط ولا المحادثة ولا اللوجات، ويكون محفوظ للتبويب الحالي فقط.",
                RequestCapability.PROJECT_STATE,
            ),
            (
                "لو العملية ماتت أثناء model_running لا تعيد تشغيل النموذج بشكل أعمى.",
                RequestCapability.PROJECT_STATE,
            ),
            (
                "نظف بيانات الاختبارات لكن لا تكسر audit chain ولا تحذف audit events.",
                RequestCapability.PROJECT_STATE,
            ),
            (
                "لا أريد alert في كل polling cycle، ونبّه فقط عند تغير الحالة الحقيقي.",
                RequestCapability.PROJECT_STATE,
            ),
            (
                "الـ22 contract_seed مش أسئلة استخدام حقيقية، فلا تعتبرها quality score.",
                RequestCapability.PROJECT_STATE,
            ),
            (
                "هل نضيف LangGraph أو Multi-Agent أو MCTS الآن؟",
                RequestCapability.RUNTIME_ARCHITECTURE,
            ),
        )

        for message, expected_capability in cases:
            with self.subTest(message=message):
                route = classify_request(message)
                self.assertEqual(route.capability, expected_capability)
                self.assertTrue(route.required_capabilities)
                self.assertTrue(route.abstain_if_evidence_missing)

    def test_english_terms_do_not_match_inside_larger_tokens(self) -> None:
        route = classify_request("احتفظ باسم checkpoint كما هو")
        self.assertEqual(route.capability, RequestCapability.GENERAL)
        self.assertEqual(route.required_capabilities, ())

    def test_template_ready_comparison_uses_web_search_before_source_status(self) -> None:
        route = classify_request(
            "قارن إذا أكمل بناء Ahmed Agent أو أستخدم حل/Template جاهز."
        )
        self.assertEqual(route.capability, RequestCapability.EXTERNAL_WEB_RESEARCH)
        self.assertEqual(
            route.required_capabilities,
            ("inspect_architecture_evidence", "web_search"),
        )

    def test_project_state_questions_route_to_project_evidence(self) -> None:
        route = classify_request("راجع حالة مشروع Ahmed Agent وما تم إنجازه")
        self.assertEqual(route.capability, RequestCapability.PROJECT_STATE)
        self.assertIn("inspect_runtime_evidence", route.required_capabilities)

    def test_runtime_and_architecture_questions_route_to_evidence_core(self) -> None:
        route = classify_request("ما هو runtime الفعلي وهل المعمارية الحالية runnable؟")
        self.assertEqual(route.capability, RequestCapability.RUNTIME_ARCHITECTURE)
        self.assertIn("inspect_runtime_evidence", route.required_capabilities)
        self.assertIn("inspect_architecture_evidence", route.required_capabilities)

    def test_project_sources_and_approved_decisions_use_fixed_evidence(self) -> None:
        route = classify_request("ما هي مصادر المشروع والقرار المعتمد؟")

        self.assertEqual(route.capability, RequestCapability.PROJECT_STATE)
        self.assertEqual(
            route.required_capabilities,
            ("inspect_runtime_evidence", "inspect_architecture_evidence"),
        )
        self.assertTrue(route.abstain_if_evidence_missing)

    def test_source_status_questions_use_bounded_source_status(self) -> None:
        route = classify_request("افحص حالة search provider وpage fetcher والجاهز منهما")
        self.assertEqual(route.capability, RequestCapability.SOURCE_STATUS)
        self.assertEqual(
            route.required_capabilities,
            ("inspect_source_status",),
        )

    def test_uploaded_file_questions_stay_in_my_files(self) -> None:
        route = classify_request("قارن الملف الرئيسي مع أدلة الحوادث المرفوعة")
        self.assertEqual(route.capability, RequestCapability.MY_FILES)
        self.assertIn("search_my_files", route.required_capabilities)
        self.assertNotIn("web_search", route.required_capabilities)

    def test_explicit_web_research_requires_real_web_search(self) -> None:
        route = classify_request("ابحث على الويب عن أحدث توثيق وقارن المصادر")
        self.assertEqual(route.capability, RequestCapability.EXTERNAL_WEB_RESEARCH)
        self.assertIn("web_search", route.required_capabilities)

    def test_missing_evidence_requires_explicit_abstention(self) -> None:
        route = classify_request("هل تم حفظ الملاحظات؟ تحقق قبل أن تؤكد")
        self.assertTrue(route.abstain_if_evidence_missing)
        self.assertNotIn("web_search", route.required_capabilities)


if __name__ == "__main__":
    unittest.main()