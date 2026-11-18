"""无 GPU 回归检查：候选目录隔离、类内逐类别输出、类间搜索。"""
import ast
import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd


def load_search_module():
    # 只替代 GPU 生成器导入；执行原始搜索代码，无需安装 diffusers 或加载模型。
    generator = ModuleType("tools_generator")
    generator.load_generator_model = Mock(side_effect=AssertionError("Unexpected model load"))
    generator.get_images = Mock(side_effect=AssertionError("Unexpected image generation"))
    spec = importlib.util.spec_from_file_location(
        "multi_fragment_under_test", Path(__file__).with_name("experiment_multi_fragment.py")
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", tools_generator=generator):
        spec.loader.exec_module(module)
    return module


e = load_search_module()
# 导入不会执行主程序中的运行配置，测试时显式设置相似度类别。
e.CROSS_SIMILARITY_TYPE = "clip"


class ModifierSearchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.standard = self.root / "processed"
        self.scores = self.standard
        self.standard.mkdir()
        self.filename = "sd15_object_detection_yolo11.csv"
        self.prompt = "a test object"
        self.categories = ("reference_anchor", "rendering_medium", "perceptual_cue", "affective_atmosphere", "contextual_frame")
        rows = [
            dict(base_prompt=self.prompt, modifier=f"{category}_{i}",
                 modifier_category=category, modifier_family="mechanism", score=i / 20)
            for category in self.categories for i in range(12)
        ]
        rows.extend([
            dict(base_prompt="another prompt", modifier="wrong_prompt",
                 modifier_category=self.categories[0], modifier_family="mechanism", score=1.0),
            dict(base_prompt=self.prompt, modifier="effect_only",
                 modifier_category="legacy", modifier_family="effect", score=0.9),
        ])
        pd.DataFrame(rows).to_csv(self.standard / self.filename, index=False)
        patcher = patch.object(e, "PROCESSED_MODIFIER_RESULTS_DIR", self.standard)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.kwargs = dict(
            csv_path=self.filename, candidate_type="mechanism",
            base_prompt=self.prompt, prompt_temple={}, pipe=None, seed=2026,
            num_images=1, task="object_detection", target_model="yolo11", model=None,
        )

    def test_candidates_and_category_counts(self):
        for family, count in (("mechanism", 8), ("effect", 6)):
            with self.subTest(candidate_type=family):
                inner = e.get_candidates(family, False)
                cross = e.get_candidates(family, True)
                self.assertEqual(len(inner["candidates"]), count)
                self.assertEqual((inner["min_num"], inner["max_num"]), (2, 4))
                self.assertEqual((cross["min_num"], cross["max_num"]), (2, count))
                self.assertEqual(inner["candidates"], cross["candidates"])

    def test_type_selects_directory_and_filters_rows(self):
        pool = e.load_single_modifier_scores(self.filename, self.prompt, "mechanism")
        self.assertEqual({c: len(items) for c, items in pool.items()},
                         dict.fromkeys(self.categories, 12))
        effect = e.load_single_modifier_scores(self.filename, self.prompt, "effect")
        self.assertEqual(effect["legacy"][0]["modifier"], "effect_only")
        explicit = e.load_single_modifier_scores(self.standard / self.filename, self.prompt, "mechanism")
        self.assertEqual(pool, explicit)

    def test_missing_score_file_is_reported(self):
        missing_dir = self.root / "missing"
        with patch.object(e, "PROCESSED_MODIFIER_RESULTS_DIR", missing_dir):
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                e.load_single_modifier_scores(self.filename, self.prompt, "mechanism")

    def test_cross_pool_never_supplements_or_rewrites_score_tables(self):
        path = self.scores / self.filename
        before = path.read_bytes()
        with patch.object(e, "get_scores", side_effect=AssertionError("Unexpected supplement")):
            pool, calls = e.cross_prepare_pool(**self.kwargs)
        self.assertEqual(calls, 0)
        self.assertEqual(set(pool), set(self.categories))
        self.assertEqual(before, path.read_bytes())

    def test_standard_pools_reuse_scores_without_supplementing_missing_modifiers(self):
        path = self.standard / self.filename
        for family in ("mechanism", "effect"):
            with self.subTest(candidate_type=family):
                pd.DataFrame([
                    dict(base_prompt=self.prompt, modifier=category + "_existing",
                         modifier_category=category, modifier_family=family, score=0.3)
                    for category in ("reference_anchor", "rendering_medium")
                ]).to_csv(path, index=False)
                before = path.read_bytes()
                with patch.object(e, "extra_mecha", {"reference_anchor": ["missing_modifier"]}), \
                     patch.object(e, "cross_evaluate", side_effect=AssertionError("Unexpected supplement")):
                    pool, calls = e.cross_prepare_pool(
                        **{**self.kwargs, "candidate_type": family}
                    )
                self.assertEqual(calls, 0)
                self.assertEqual(pool, {
                    "reference_anchor": [("reference_anchor_existing", 0.3)],
                    "rendering_medium": [("rendering_medium_existing", 0.3)],
                })
                self.assertEqual(path.read_bytes(), before)

    def test_mechanism_optimization_generates_combinations_with_selected_pipeline(self):
        path = self.standard / self.filename
        pd.DataFrame([
            dict(base_prompt=self.prompt, modifier=category + "_existing",
                 modifier_category=category, modifier_family="mechanism", score=0.3)
            for category in ("reference_anchor", "rendering_medium")
        ]).to_csv(path, index=False)
        before = path.read_bytes()
        selected_pipe = object()
        combo = {"reference_anchor": "reference_anchor_existing", "rendering_medium": "rendering_medium_existing"}
        with patch.object(e, "cross_initial_combos", return_value=[combo]), \
             patch.object(e, "extra_mecha", {"reference_anchor": ["missing_modifier"]}), \
             patch.object(e, "get_scores", return_value=[1.0]) as score, \
             patch.object(e, "cross_image_text_similarity", return_value=[25.0]):
            result = e.optimize_cross_true(
                **{**self.kwargs, "candidate_type": "mechanism", "pipe": selected_pipe},
                total_budget=500, output_dir=self.root / "selected_generator",
                generator_model="sld-medium",
            )
        score.assert_called_once()
        self.assertIs(score.call_args.args[2], selected_pipe)
        self.assertEqual(score.call_args.args[0], self.prompt + ", reference_anchor_existing, rendering_medium_existing")
        self.assertEqual(result["image_generation_calls"], 1)
        self.assertEqual(path.read_bytes(), before)

    def test_inner_search_uses_top_k_within_each_category(self):
        with patch.object(e, "get_scores", return_value=[0.5]) as score:
            results = e.optimize_cross_false(**self.kwargs, top_k=2)
        self.assertEqual([r["category"] for r in results], list(self.categories))
        self.assertEqual([r["image_generation_calls"] for r in results], [1] * 5)
        for result in results:
            category = result["category"]
            self.assertEqual(result["best_modifier"], [f"{category}_11", f"{category}_10"])
            self.assertEqual(result["best_score"], 0.5)
        self.assertEqual(score.call_count, 5)
        actual = {call.args[0] for call in score.call_args_list}
        expected = {f"{self.prompt}, {c}_11, {c}_10" for c in self.categories}
        self.assertEqual(actual, expected)

    def test_inner_perfect_score_stops_only_current_category(self):
        # 第一类首个组合即满分；其余类别仍须各自评估四个组合并选出最优。
        scores = [[1.0]] + [[0.4], [0.9], [0.6], [0.8]] * 4
        with patch.object(e, "get_scores", side_effect=scores) as score:
            results = e.optimize_cross_false(**self.kwargs, top_k=3)
        self.assertEqual(score.call_count, 17)
        self.assertEqual([r["category"] for r in results], list(self.categories))
        self.assertEqual([r["image_generation_calls"] for r in results], [1, 4, 4, 4, 4])
        self.assertEqual([r["best_score"] for r in results], [1.0, 0.9, 0.9, 0.9, 0.9])
        for result in results[1:]:
            category = result["category"]
            self.assertEqual(result["best_modifier"], [f"{category}_11", f"{category}_9"])

    def test_inner_insufficient_candidates_identifies_category(self):
        with self.assertRaisesRegex(ValueError, "category='reference_anchor'"):
            e.optimize_cross_false(**self.kwargs, top_k=1)

    def test_inner_main_csv_has_one_row_per_prompt_and_category(self):
        # 执行实际主入口，替代模型和输入数据；验证最终CSV列序与提示词覆盖。
        source = Path(e.__file__)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "__name__ == '__main__'")
        inner_source = next(
            node.value.value
            for node in main.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and "# 类内" in node.value.value
        )
        inner_source = textwrap.dedent("\n".join(inner_source.splitlines()[1:]))
        inner_source = (
            "args = parse_args()\n"
            "device = args.device\n"
            "IMAGE_OUTPUT_DIR = args.path\n"
            "generator_model = args.model\n"
            "torch.cuda.set_device(device)\n"
            + inner_source
        )
        code = compile(inner_source, str(source), "exec")
        prompts = [dict(task="object_detection", base_prompt=text)
                   for text in ("first prompt", "second prompt")]
        pool = {category: [{"modifier": f"{category}_{i}", "score": i / 20}
                           for i in range(3)] for category in self.categories}
        output_dir = self.root / "multi_fragment"
        generator_loader = Mock(return_value=None)
        target_loader = Mock(return_value=None)
        with patch.object(e, "get_scores", return_value=[1.0]) as score, \
             patch.object(e, "load_single_modifier_scores", return_value=pool), \
             patch.object(e.torch.cuda, "set_device") as set_device, \
             patch("builtins.print"):
            namespace = dict(vars(e))
            namespace.update(
                MULTI_FRAGMENT_RESULTS_DIR=output_dir,
                get_prompt_temples=lambda task: prompts,
                parse_args=lambda: SimpleNamespace(device="cuda:1", path="./img/img1", model="sd15"),
                load_generator_model=generator_loader,
                load_target_model=target_loader,
            )
            exec(code, namespace)
        self.assertEqual(namespace["IMAGE_OUTPUT_DIR"], "./img/img1")
        set_device.assert_called_once_with("cuda:1")
        generator_loader.assert_called_once_with("sd15", device="cuda:1", cpu_offload=False)
        self.assertEqual(target_loader.call_count, len(prompts))
        self.assertTrue(all(call.kwargs["device"] == "cuda:1"
                            for call in target_loader.call_args_list))
        paths = list(output_dir.glob("*_sd15_inner.csv"))
        self.assertEqual(len(paths), 1)
        frame = pd.read_csv(paths[0])
        self.assertEqual(list(frame.columns), [
            "category", "base_prompt", "best_modifier", "best_score",
            "best_prompt", "image_generation_calls",
        ])
        self.assertEqual(len(frame), 10)
        self.assertEqual(score.call_count, 10)
        self.assertEqual(set(zip(frame.base_prompt, frame.category)),
                         {(p["base_prompt"], c) for p in prompts for c in self.categories})
        self.assertTrue((frame.image_generation_calls == 1).all())

    def test_device_command_line(self):
        self.assertEqual(e.parse_args([]).model, "sd15")
        self.assertEqual(e.parse_args(["--model", "flux.1-dev"]).model, "flux.1-dev")
        self.assertEqual(e.parse_args([]).device, "cuda:0")
        self.assertEqual(e.parse_args(["--device", "cuda:1"]).device, "cuda:1")
        self.assertEqual(e.parse_args(["--device", "cuda"]).device, "cuda:0")
        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            e.parse_args(["--device", "cuda:-1"])
        self.assertEqual(error.exception.code, 2)

    def test_image_path_command_line(self):
        self.assertEqual(e.parse_args([]).path, "./img/img")
        args = e.parse_args(["--device", "cuda:1", "--path", "./img/img1"])
        self.assertEqual(args.path, "./img/img1")
        self.assertEqual(args.device, "cuda:1")

    def test_get_scores_uses_same_image_path_for_generation_and_evaluation(self):
        for path in ("./img/img", "./img/img1"):
            with self.subTest(path=path), \
                 patch.object(e, "IMAGE_OUTPUT_DIR", path), \
                 patch.object(e, "generate_images") as generate, \
                 patch.object(e, "run_target_model", return_value=[]) as target, \
                 patch.object(e, "get_eval_prompt_fields", return_value={}), \
                 patch.object(e, "get_eval_model_fields", return_value=[]), \
                 patch.object(e, "evaluate_scores", return_value=[0.5]):
                result = e.get_scores("test", {}, None, 2026, 1,
                                      "image_classification", "mambavision", None)
            generate.assert_called_once_with(None, "test", 2026, 1, output_dir=path)
            target.assert_called_once_with(None, "image_classification", "mambavision",
                                           {}, image_dir=path)
            self.assertEqual(result, [0.5])

    def test_device_forwarded_to_target_loaders(self):
        cases = [
            ("image_classification", "mambavision", "load_image_classifier"),
            ("image_classification", "resnet50", "load_image_classifier"),
            ("object_detection", "yolo11", "load_object_extractor"),
            ("object_detection", "deim", "load_object_extractor"),
            ("segmentation", "sam2", "load_segmenter"),
            ("segmentation", "deeplabv3plus", "load_segmenter"),
            ("visual_understanding", "qwen3-vl", "load_visual_understander"),
            ("visual_understanding", "llava-1.5", "load_visual_understander"),
        ]
        for task, name, loader in cases:
            with self.subTest(task=task, name=name), patch.object(e, loader) as mocked:
                e.load_target_model(task, name, device="cuda:1")
                self.assertEqual(mocked.call_args.kwargs["device"], "cuda:1")

    def test_cross_search_runs_all_stages_and_saves_separate_history(self):
        before = (self.scores / self.filename).read_bytes()
        history_dir = self.root / "history"
        with patch.object(e, "get_scores", return_value=[0.5]) as score, \
             patch.object(e, "cross_image_text_similarity",
                          return_value=[30.0]):
            result = e.optimize_cross_true(
                **self.kwargs, total_budget=170, output_dir=history_dir
            )
        self.assertEqual(result["image_generation_calls"], score.call_count)
        self.assertLessEqual(score.call_count, 170)
        self.assertTrue(2 <= len(result["best_modifier"]) <= 5)
        history = pd.read_csv(
            history_dir / "sd15_object_detection_yolo11_mechanism_cross_history.csv"
        )
        self.assertEqual(set(history.stage),
                         {"initial", "local_refine", "final_repeat"}
                         | {f"surrogate_{i}" for i in range(1, 6)})
        self.assertEqual(len(history), score.call_count)
        self.assertFalse(history.prompt.str.contains("wrong_").any())
        self.assertTrue((history.sim == 30.0).all())
        self.assertAlmostEqual(result["best_score"], 0.5)
        self.assertEqual(result["best_score_sequence"], [0.5])
        self.assertEqual(result["sim_sequence"], [30.0])
        self.assertEqual(before, (self.scores / self.filename).read_bytes())

    def test_cross_full_score_only_stops_after_clip_threshold_passes(self):
        history_dir = self.root / "history_threshold"
        similarities = [[24.0], [26.0]]
        with patch.object(e, "get_scores", return_value=[1.0]) as score, \
             patch.object(e, "cross_image_text_similarity",
                          side_effect=similarities):
            result = e.optimize_cross_true(
                **self.kwargs, total_budget=170, output_dir=history_dir
            )
        self.assertEqual(score.call_count, 2)
        self.assertEqual(result["best_score"], 1.0)
        self.assertEqual(result["sim"], 26.0)
        self.assertEqual(result["sim_sequence"], [26.0])
        self.assertEqual(result["best_score_sequence"], [1.0])
        history = pd.read_csv(
            history_dir / "sd15_object_detection_yolo11_mechanism_cross_history.csv"
        )
        self.assertEqual(history.sim.tolist(), [24.0, 26.0])
        self.assertEqual(history.res_loss.tolist(), [0.96, 1.0])
        self.assertEqual(history.score.tolist(), [1.0, 1.0])
        self.assertEqual(result["res_loss_sequence"], [1.0])

    def test_cross_joint_formula_is_per_image(self):
        for threshold in (25.0, 70.0):
            with self.subTest(threshold=threshold):
                actual = e.cross_joint_scores(
                    [0.8, 0.6, 0.4, 1.0],
                    [0.8 * threshold, threshold, threshold + 1, 0.0],
                    threshold,
                )
                e.np.testing.assert_allclose(actual, [0.64, 0.6, 0.4, 0.0])
        with self.assertRaisesRegex(ValueError, "equal nonzero lengths"):
            e.cross_joint_scores([0.5, 0.6], [30.0], 25.0)

    def test_cross_all_low_similarity_still_returns_best_joint_score(self):
        # 两张图的联合分数分别为0.64和0.6，均值0.62；不得再硬过滤。
        with patch.object(e, "get_scores", return_value=[0.8, 0.6]), \
             patch.object(e, "cross_image_text_similarity", return_value=[20.0, 25.0]):
            evaluation = e.cross_evaluate(
                {"reference_anchor": "test"}, self.prompt, {}, None, 2026, 2,
                "object_detection", "yolo11", None,
            )
        e.np.testing.assert_allclose(evaluation["scores"], [0.64, 0.6])
        self.assertAlmostEqual(evaluation["score"], 0.62)
        self.assertEqual(evaluation["task_scores"], [0.8, 0.6])
        self.assertEqual(evaluation["sim_sequence"], [20.0, 25.0])

        kwargs = {**self.kwargs, "num_images": 2}
        with patch.object(e, "get_scores", return_value=[0.8, 0.6]), \
             patch.object(e, "cross_image_text_similarity", return_value=[20.0, 20.0]):
            result = e.optimize_cross_true(
                **kwargs, total_budget=170, output_dir=self.root / "joint_history"
            )
        self.assertAlmostEqual(result["res_loss"], 0.56)
        e.np.testing.assert_allclose(result["res_loss_sequence"], [0.64, 0.48])
        self.assertAlmostEqual(result["best_score"], 0.7)
        e.np.testing.assert_allclose(result["best_score_sequence"], [0.8, 0.6])
        self.assertEqual(result["sim_sequence"], [20.0, 20.0])
        self.assertEqual(result["sim"], 20.0)

    def test_cross_ranks_joint_score_instead_of_raw_task_score(self):
        first = {"reference_anchor": "reference_anchor_0", "rendering_medium": "rendering_medium_0"}
        second = {"reference_anchor": "reference_anchor_1", "rendering_medium": "rendering_medium_1"}
        with patch.object(e, "cross_initial_combos", return_value=[first, second]), \
             patch.object(e, "cross_surrogate_search", return_value=0), \
             patch.object(e, "cross_refine_and_verify", return_value=0), \
             patch.object(e, "get_scores", side_effect=[[0.9], [0.8]]), \
             patch.object(e, "cross_image_text_similarity", side_effect=[[10.0], [25.0]]):
            result = e.optimize_cross_true(
                **self.kwargs, total_budget=170, output_dir=self.root / "ranking",
                generator_model="sld-medium",
            )
        self.assertEqual(result["best_modifier"], second)
        self.assertEqual(result["best_score_sequence"], [0.8])
        self.assertEqual(result["sim_sequence"], [25.0])
        self.assertTrue((self.root / "ranking/sld-medium_object_detection_yolo11_mechanism_cross_history.csv").is_file())
        self.assertFalse((self.root / "ranking/sd15_object_detection_yolo11_mechanism_cross_history.csv").exists())

    def test_cross_main_csv_adds_only_requested_metrics(self):
        source = Path(e.__file__)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "__name__ == '__main__'")
        code = compile(ast.Module(body=main.body, type_ignores=[]), str(source), "exec")
        result = dict(best_modifier={"reference_anchor": "a", "rendering_medium": "b"},
                      best_prompt="test, a, b", res_loss=0.64,
                      res_loss_sequence=[0.64], best_score=0.8,
                      best_score_sequence=[0.8], sim=20.0, sim_sequence=[20.0],
                      image_generation_calls=1)
        for generator_model in ('sd15', 'sld-medium', 'flux.1-dev'):
            with self.subTest(generator_model=generator_model):
                csv_path = self.standard / "sd15_image_classification_mambavision.csv"
                csv_path.write_text('fixture')
                output_dir = self.root / generator_model
                namespace = dict(vars(e))
                namespace.update(
                    parse_args=lambda: SimpleNamespace(device="cuda:1", path="./img/img1",
                                                      model=generator_model, task="image_classification",
                                                      target_model="mambavision", candidate_type="mechanism"),
                    load_generator_model=Mock(), load_target_model=Mock(),
                    get_prompt_temples=lambda task: [dict(task=task, base_prompt="test")],
                    optimize_cross_true=Mock(return_value=result),
                    MULTI_FRAGMENT_RESULTS_DIR=output_dir,
                )
                with patch.object(e.torch.cuda, "set_device"), patch("builtins.print"):
                    exec(code, namespace)
                namespace['load_generator_model'].assert_called_once_with(
                    generator_model, device='cuda:1', cpu_offload=generator_model == 'flux.1-dev')
                optimization = namespace['optimize_cross_true'].call_args.kwargs
                self.assertEqual(optimization['csv_path'], csv_path)
                self.assertEqual(optimization['generator_model'], generator_model)
                self.assertIs(optimization['pipe'], namespace['load_generator_model'].return_value)
                self.assertEqual(optimization['total_budget'], 500)
                frame = pd.read_csv(output_dir / f"image_classification_mambavision_mechanism_2026_{generator_model}_cross.csv")
                self.assertEqual(list(frame.columns), [
                    "base_prompt", "best_modifier", "res_loss", "res_loss_sequence",
                    "best_score", "best_score_sequence", "sim", "sim_sequence",
                    "best_prompt", "image_generation_calls",
                ])
                for mean_col, seq_col in (("res_loss", "res_loss_sequence"),
                                          ("best_score", "best_score_sequence"),
                                          ("sim", "sim_sequence")):
                    sequence = ast.literal_eval(frame.iloc[0][seq_col])
                    self.assertEqual(len(sequence), 1)
                    self.assertAlmostEqual(frame.iloc[0][mean_col], sum(sequence) / len(sequence))

    def test_missing_shared_sd15_score_file_fails_before_loading_model(self):
        source = Path(e.__file__)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "__name__ == '__main__'")
        code = compile(ast.Module(body=main.body, type_ignores=[]), str(source), "exec")
        baseline = self.standard / self.filename
        before = baseline.read_bytes()
        loader = Mock()
        namespace = dict(vars(e))
        namespace.update(
            parse_args=lambda: SimpleNamespace(device='cuda:0', path='./img/img',
                                              model='flux.1-dev', task='object_detection',
                                              target_model='deim', candidate_type='mechanism'),
            load_generator_model=loader, MULTI_FRAGMENT_RESULTS_DIR=self.root/'missing_model',
        )
        with patch.object(e.torch.cuda, 'set_device'), self.assertRaisesRegex(
                FileNotFoundError, 'sd15_object_detection_deim.csv'):
            exec(code, namespace)
        loader.assert_not_called()
        self.assertEqual(baseline.read_bytes(), before)

    def test_cross_repeats_average_each_metric_independently(self):
        combo = {"reference_anchor": "reference_anchor_0", "rendering_medium": "rendering_medium_0"}
        def repeat(*args, **kwargs):
            return e.cross_test_combos(
                [combo], args[1], args[2], args[3], args[4], args[5],
                args[6], args[7], args[8], args[9], args[10],
                stage="final_repeat", allow_repeat=True,
            )
        with patch.object(e, "cross_initial_combos", return_value=[combo]), \
             patch.object(e, "cross_surrogate_search", return_value=0), \
             patch.object(e, "cross_refine_and_verify", side_effect=repeat), \
             patch.object(e, "get_scores", side_effect=[[1.0, 0.4], [0.6, 0.8]]), \
             patch.object(e, "cross_image_text_similarity", side_effect=[[10.0, 25.0], [25.0, 0.0]]):
            result = e.optimize_cross_true(
                **{**self.kwargs, "num_images": 2}, total_budget=170,
                output_dir=self.root / "repeated_metrics",
            )
        e.np.testing.assert_allclose(result["res_loss_sequence"], [0.5, 0.2])
        e.np.testing.assert_allclose(result["best_score_sequence"], [0.8, 0.6])
        e.np.testing.assert_allclose(result["sim_sequence"], [17.5, 12.5])
        self.assertAlmostEqual(result["res_loss"], 0.35)
        self.assertAlmostEqual(result["best_score"], 0.7)
        self.assertAlmostEqual(result["sim"], 15.0)

    def test_cross_similarity_type_switches_between_clip_and_blip(self):
        image_dir = self.root / "similarity_images"
        image_dir.mkdir()
        (image_dir / "0.png").write_bytes(b"mock")
        (image_dir / "1.png").write_bytes(b"mock")
        evaluation_module = ModuleType("tools_evaluation")
        calculate = Mock(side_effect=[75.0, 80.0])
        evaluation_module.similarity_calculate = calculate
        with patch.object(e, "IMAGE_OUTPUT_DIR", str(image_dir)), \
             patch.object(e, "CROSS_SIMILARITY_TYPE", "blip"), \
             patch.dict("sys.modules", tools_evaluation=evaluation_module):
            scores = e.cross_image_text_similarity(self.prompt)
        self.assertEqual(scores, [75.0, 80.0])
        self.assertEqual(
            [call.args[0] for call in calculate.call_args_list],
            ["BLIP", "BLIP"],
        )
        self.assertEqual(e.CROSS_SIMILARITY_CONFIG["blip"]["threshold"], 70.0)
        with patch.object(e, "CROSS_SIMILARITY_TYPE", "blip"), \
             patch.object(e, "get_scores", return_value=[1.0]), \
             patch.object(e, "cross_image_text_similarity",
                          return_value=[69.0]):
            evaluation = e.cross_evaluate(
                {"reference_anchor": "test"}, self.prompt, {}, None, 2026, 1,
                "object_detection", "yolo11", None,
            )
        self.assertAlmostEqual(evaluation["score"], 69.0 / 70)
        self.assertEqual(evaluation["scores"], [69.0 / 70])
        with patch.object(e, "CROSS_SIMILARITY_TYPE", "unknown"):
            with self.assertRaisesRegex(ValueError, "CROSS_SIMILARITY_TYPE"):
                e.cross_similarity_config()


if __name__ == "__main__":
    unittest.main()
