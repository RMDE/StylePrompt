"""按三种模式顺序运行类间实验；验证结果完整性，异常退出时保留输出供检查。"""
import argparse
from collections import Counter
import csv
import importlib.util
from datetime import datetime
from functools import lru_cache
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENT_SCRIPT = PROJECT_DIR / "experiment_multi_fragment.py"
RESULTS_DIR = PROJECT_DIR / "experiment_outputs" / "multi_fragment"


def default_environment_python(name):
    """优先使用当前同名环境，再查找同一Conda安装和已登记的环境。"""
    candidates = []
    prefixes = [Path(sys.prefix)]
    if os.environ.get("CONDA_PREFIX"):
        prefixes.append(Path(os.environ["CONDA_PREFIX"]))
    for prefix in prefixes:
        if prefix.name == name:
            candidates.append(prefix / "bin/python")
    roots = []
    for prefix in prefixes:
        roots.append(prefix.parent.parent if prefix.parent.name == "envs" else prefix)
    for executable in [os.environ.get("CONDA_EXE"), os.environ.get("CONDA_PYTHON_EXE"), shutil.which("conda")]:
        if executable:
            roots.append(Path(executable).parent.parent)
    for root in roots:
        candidates.append(root / "envs" / name / "bin/python")
    for directory in os.environ.get("CONDA_ENVS_PATH", "").split(os.pathsep):
        if directory:
            candidates.append(Path(directory) / name / "bin/python")
    try:
        for line in (Path.home() / ".conda/environments.txt").read_text().splitlines():
            prefix = Path(line.strip())
            if prefix.name == name:
                candidates.append(prefix / "bin/python")
    except OSError:
        pass
    for distribution in ("anaconda3", "miniconda3", "miniforge3", "mambaforge"):
        candidates.append(Path.home() / distribution / "envs" / name / "bin/python")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    # 仍可通过--cv-python/--nsfw-python/--python显式指定；执行前会给出明确诊断。
    return Path.home() / "miniconda3/envs" / name / "bin/python"


DEFAULT_CV_PYTHON = default_environment_python("cvTask")
DEFAULT_NSFW_PYTHON = default_environment_python("styleprompt3")
TASK_MODELS = {
    "image_classification": ("resnet50", "mambavision"),
    "object_detection": ("yolo11", "deim"),
    "segmentation": ("deeplabv3plus", "sam2"),
    "nsfw": ("clip", "nudenet", "q16"),
    "visual_understanding": ("qwen3-vl", "llava-1.5"),
}
MODE_DESCRIPTIONS = {
    "5090d": "除visual_understanding外的全部任务（18组）",
    "4090d": "仅图像分类MambaVision的两类候选实验（2组）",
    "L40": "仅visual_understanding两个目标模型的两类候选实验（4组）",
}
# 与 experiment_multi_fragment.py 的类间入口保持一致。
SEED = 2026
GENERATOR_MODELS = ("sd15", "sdxl", "sd3", "sld-medium", "sld-strong", "sld-max",
                    "flux.1-dev", "flux-3-image-klein")
SLD_MODELS = ("sld-medium", "sld-strong", "sld-max")


def mode_targets(mode):
    if mode == "5090d":
        return {task: targets for task, targets in TASK_MODELS.items()
                if task != "visual_understanding"}
    if mode == "4090d":
        return {"image_classification": ("mambavision",)}
    if mode == "L40":
        return {"visual_understanding": TASK_MODELS["visual_understanding"]}
    raise ValueError("mode 必须为 5090d、4090d 或 L40")


def experiment_jobs(tasks=None, model="sd15", mode="5090d"):
    targets = mode_targets(mode)
    selected = list(targets) if tasks is None else list(dict.fromkeys(tasks))
    invalid = [task for task in selected if task not in targets]
    if invalid:
        raise ValueError(f"模式{mode}不包含任务：{', '.join(invalid)}；可选任务：{', '.join(targets)}")
    for task in selected:
        if model in SLD_MODELS and task != "nsfw":
            continue
        for candidate_type in ("mechanism", "effect"):
            for target_model in targets[task]:
                yield task, candidate_type, target_model


def output_csv_paths(results_dir, task, candidate_type, target_model, model="sd15"):
    return (
        results_dir / f"{task}_{target_model}_{candidate_type}_{SEED}_{model}_cross.csv",
        results_dir / f"{model}_{task}_{target_model}_{candidate_type}_cross_history.csv",
    )


@lru_cache(maxsize=None)
def expected_base_prompts(task, target_model):
    # 只导入提示词定义，批量调度与完整性检查均不加载torch或视觉模型。
    from test_prompts_design import (
        build_image_classification_prompts, build_object_detection_prompts,
        SEGMENTATION_PROMPT_GROUPS, VISUAL_UNDERSTANDING_PROMPTS, iter_prompt_jobs,
    )
    if task == "image_classification":
        return tuple(p["base_prompt"] for p in build_image_classification_prompts())
    if task == "object_detection":
        return tuple(p["base_prompt"] for p in build_object_detection_prompts())
    if task == "segmentation":
        return tuple(p["prompt"] for g in SEGMENTATION_PROMPT_GROUPS for p in g["prompts"])
    if task == "visual_understanding":
        return tuple(p["base_prompt"] for p in VISUAL_UNDERSTANDING_PROMPTS)
    if task == "nsfw":
        subtype = {"clip": "sex", "nudenet": "sex", "q16": "violence"}[target_model]
        return tuple(prompt for label, _, _, prompt in iter_prompt_jobs() if label == subtype)
    raise ValueError(f"未知任务：{task}")


def check_result_file(path, task, target_model, test=False):
    """检查主结果CSV；历史CSV只含搜索轨迹，不用于判断任务是否完成。"""
    expected = expected_base_prompts(task, target_model)
    if test:
        expected = expected[:1]
    check = dict(complete=False, expected_rows=len(expected), existing_rows=0, reason="结果文件不存在")
    if not path.exists():
        return check
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = reader.fieldnames or []
            records = list(reader)
        check["existing_rows"] = len(records)
        required = {"base_prompt", "best_modifier", "best_score", "best_prompt"}
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            check["reason"] = "缺少结果必需字段或表头重复"
        elif len(records) != len(expected):
            check["reason"] = f"记录条数{len(records)}，预期{len(expected)}"
        elif any(None in row or any(row.get(k) is None for k in fields) for row in records):
            check["reason"] = "存在列数不匹配或截断记录"
        elif Counter(row["base_prompt"] for row in records) != Counter(expected):
            check["reason"] = "基础提示词缺失、重复或与当前定义不一致"
        elif any(not row[k].strip() or row[k].strip().lower() in {"nan", "null", "none"}
                 for row in records for k in required):
            check["reason"] = "存在未填写的结果字段"
        elif any(not math.isfinite(float(row["best_score"])) for row in records):
            check["reason"] = "存在无效best_score"
        else:
            check.update(complete=True, reason=f"结果完整（{len(records)}/{len(expected)}条）")
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        check["reason"] = f"结果无法读取或内容无效：{exc}"
    return check


def cleanup_csvs(paths):
    errors = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    return errors


def run_streaming_command(command, *, stdout, live=True):
    """始终保存完整输出；live=True时同时实时显示到终端。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
    with subprocess.Popen(command, cwd=PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1, env=env) as process:
        try:
            for line in process.stdout:
                if live:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                stdout.write(line)
                stdout.flush()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        return subprocess.CompletedProcess(command, process.wait())


def show_failure_details(path):
    """正常模式在退出后展示完整异常，不展示此前的加载/生成进度。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = next((i for i, line in enumerate(lines)
                      if "Traceback (most recent call last)" in line or "Fatal Python error:" in line), None)
        print(f"错误日志：{path}", file=sys.stderr, flush=True)
        if start is not None:
            print("\n".join(lines[start:]), file=sys.stderr, flush=True)
        else:
            errors = [line for line in lines if re.search(
                r"(^|\s)(ERROR|FATAL)\b|^\w*(?:Error|Exception)(?::|\b)", line)]
            print("\n".join(errors) if errors else "子进程未输出可识别的异常信息，请查看完整日志。",
                  file=sys.stderr, flush=True)
    except OSError as exc:
        print(f"无法读取错误日志：{exc}", file=sys.stderr, flush=True)


def show_final_results(path):
    print(f"结果文件：{path}", flush=True)
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream), 1):
                print(f"[{index}] base_prompt: {row['base_prompt']}", flush=True)
                for field in ("best_modifier", "best_score", "res_loss", "sim", "best_prompt"):
                    if field in row:
                        print(f"{field}: {row[field]}", flush=True)
    except (OSError, UnicodeError, csv.Error) as exc:
        print(f"无法显示结果内容：{exc}", file=sys.stderr, flush=True)


def run_single_prompt_test(args):
    """在独立子进程中加载原实验模块，仅选首条prompt，复用完整搜索流程。"""
    spec = importlib.util.spec_from_file_location("_multi_fragment_test_experiment", args.experiment_script)
    experiment = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = experiment
    spec.loader.exec_module(experiment)
    experiment.IMAGE_OUTPUT_DIR = args.path
    experiment.CROSS_SIMILARITY_TYPE = "blip"
    experiment.torch.cuda.set_device(args.device)
    prompts = experiment.get_prompt_temples(args.task)
    if args.task == "nsfw":
        subtype = {"clip": "sex", "nudenet": "sex", "q16": "violence"}[args.target_model]
        prompts = [p for p in prompts if p["nsfw_type"] == subtype]
    prompt = prompts[0]
    print(f"测试基础提示词：{prompt['base_prompt']}", flush=True)
    csv_path = (experiment.get_single_modifier_results_dir(args.candidate_type)
                / f"sd15_{args.task}_{args.target_model}.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(f"缺少共用的SD1.5单修饰词分数CSV: {csv_path}")
    pipe = experiment.load_generator_model(args.model, device=args.device,
                                           cpu_offload=args.model == "flux.1-dev")
    model = experiment.load_target_model(args.task, args.target_model, device=args.device)
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = experiment.optimize_cross_true(
        csv_path=csv_path, candidate_type=args.candidate_type, base_prompt=prompt["base_prompt"],
        prompt_temple=prompt, pipe=pipe, seed=SEED, num_images=1, task=args.task,
        target_model=args.target_model, model=model, total_budget=500,
        output_dir=output_dir, generator_model=args.model,
    )
    row = dict(base_prompt=prompt["base_prompt"], **result)
    result_path, _ = output_csv_paths(output_dir, args.task, args.candidate_type, args.target_model, args.model)
    with result_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(result, flush=True)
    return 0


def run_batch(python, device, image_dir, log_dir, *, dry_run=False,
              script=EXPERIMENT_SCRIPT, results_dir=RESULTS_DIR, tasks=None,
              cv_python=DEFAULT_CV_PYTHON, nsfw_python=DEFAULT_NSFW_PYTHON, model="sd15", mode="5090d",
              test=False):
    results_dir = Path(results_dir)
    if test:
        results_dir = results_dir / "test_run"
        print(f"测试模式：每个实验组合仅运行第一条基础提示词，预期1条结果；保存到 {results_dir}", flush=True)
    jobs = list(experiment_jobs(tasks, model=model, mode=mode))
    print(f"模式{mode}：{MODE_DESCRIPTIONS[mode]}", flush=True)
    if model in SLD_MODELS:
        print("SLD仅运行NSFW任务，其他任务自动跳过。", flush=True)
    if not jobs:
        print("没有需要运行的任务。", flush=True)
        return 0
    # 直接复用用户指定的目录；并行批次通过不同--path进行隔离。
    image_dir = Path(image_dir).resolve()
    succeeded = failed = skipped = abnormal_exits = 0
    if not dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    for index, (task, candidate_type, target_model) in enumerate(jobs, 1):
        label = f"{index:02d}_{model}_{task}_{candidate_type}_{target_model}"
        environment = "styleprompt3" if task == "nsfw" else "cvTask"
        job_python = python or (nsfw_python if task == "nsfw" else cv_python)
        if python:
            environment = "custom"
        entry = ([str(Path(__file__).resolve()), "--test-worker", "--experiment-script", str(script),
                  "--results-dir", str(results_dir)] if test else [str(script)])
        command = [str(job_python), "-u", *entry, "--device", device,
                   "--path", str(image_dir), "--task", task,
                   "--target-model", target_model, "--candidate-type", candidate_type,
                   "--model", model]
        print(f"[{index}/{len(jobs)}] {model} / {task} / {candidate_type} / {target_model} [{environment}]", flush=True)
        output_paths = output_csv_paths(results_dir, task, candidate_type, target_model, model)
        existing = check_result_file(output_paths[0], task, target_model, test=test)
        log_path = log_dir / f"{label}.log"
        if existing["complete"]:
            print(f"{existing['reason']}，跳过。", flush=True)
            skipped += 1
            if not dry_run:
                log_path.write_text(f"跳过：{output_paths[0]}\n{existing['reason']}\n", encoding="utf-8")
            continue
        if test or dry_run:
            print(f"{existing['reason']}，从头运行。", flush=True)
            print(f"Python：{job_python}", flush=True)
        if dry_run:
            print(shlex.join(command))
            continue
        if test:
            print(f"日志：{log_path}", flush=True)
        # 从头运行前删除这一组的旧结果及历史，避免残留或追加。
        startup_error = None
        if shutil.which(str(job_python)) is None:
            option = "--nsfw-python" if task == "nsfw" else "--cv-python"
            startup_error = (f"无法启动进程：Python解释器不存在或不可执行：{job_python}。"
                             f"请使用{option}指定正确路径，或用--python指定当前环境的Python。")
        cleanup_errors = [] if startup_error else cleanup_csvs(output_paths)
        process_started = False
        with log_path.open("w", encoding="utf-8") as log:
            log.write(existing["reason"] + "\n")
            log.write(shlex.join(command) + "\n")
            log.flush()
            if startup_error:
                log.write(startup_error + "\n")
                returncode = -1
            elif cleanup_errors:
                log.write("无法清理旧CSV，未启动实验：" + "; ".join(cleanup_errors) + "\n")
                returncode = -1
            else:
                try:
                    result = run_streaming_command(command, stdout=log, live=test)
                    process_started = True
                    returncode = result.returncode
                except OSError as exc:
                    startup_error = f"无法启动进程：{exc}"
                    log.write(startup_error + "\n")
                    returncode = -1
        if not process_started:
            failed += 1
            if startup_error:
                print(startup_error, file=sys.stderr, flush=True)
                print(f"失败（退出码 {returncode}），实验未启动，继续下一组。", flush=True)
            else:
                print(f"失败（退出码 {returncode}），旧CSV清理失败，实验未启动，继续下一组。", flush=True)
            for error in cleanup_errors:
                print(f"CSV清理失败：{error}", file=sys.stderr, flush=True)
            continue
        # 进程可能在写出完整结果后才异常退出，也可能退出码为0却只写出部分结果。
        # 两种情况都以主结果CSV的完整性判定；退出码及日志仍保留用于诊断。
        finished = check_result_file(output_paths[0], task, target_model, test=test)
        if finished["complete"]:
            succeeded += 1
            if returncode != 0:
                abnormal_exits += 1
                print(f"{finished['reason']}，但进程异常退出（退出码 {returncode}）；"
                      "保留结果及历史CSV，请检查日志中的异常。", flush=True)
            else:
                print(f"完成：{finished['reason']}。", flush=True)
        else:
            failed += 1
            print(f"任务未完成（退出码 {returncode}）：{finished['reason']}。"
                  "保留现有结果及历史CSV，继续下一组；下次运行将从头重跑。", flush=True)
        if not test:
            if returncode != 0 or not finished["complete"]:
                show_failure_details(log_path)
            if finished["complete"]:
                show_final_results(output_paths[0])
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n批量检查：退出码 {returncode}；{finished['reason']}；保留输出。\n")
    if not dry_run:
        print(f"批量运行结束：成功 {succeeded}，跳过 {skipped}，失败 {failed}。")
        if abnormal_exits:
            print(f"其中 {abnormal_exits} 项结果完整但进程异常退出，详见对应日志。")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=list(MODE_DESCRIPTIONS), default="5090d",
                        help="5090d=除视觉理解外全部任务（默认）；4090d=仅MambaVision三组；L40=视觉理解全系列六组")
    parser.add_argument("--model", choices=GENERATOR_MODELS, default="sd15",
                        help="组合优化使用的生成模型；SLD仅运行NSFW；候选分数统一读取SD1.5记录")
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_MODELS), default=None,
                        help="可选：在当前模式内进一步选择任务并指定顺序")
    parser.add_argument("--cv-python", default=str(DEFAULT_CV_PYTHON),
                        help="分类、检测、分割、视觉理解使用的cvTask解释器（默认自动查找）")
    parser.add_argument("--nsfw-python", default=str(DEFAULT_NSFW_PYTHON),
                        help="NSFW使用的styleprompt3解释器（默认自动查找）")
    parser.add_argument("--python", default=None,
                        help="手动覆盖所有任务的解释器；默认按任务自动选择环境")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--path", default=str(PROJECT_DIR / "img" / "multi_fragment_batch"),
                        help="直接传给实验脚本的固定图像目录，不追加子目录；每轮清空复用，并行进程请指定不同路径")
    parser.add_argument("--dry-run", action="store_true", help="仅打印选中任务的命令，不运行或删除文件")
    parser.add_argument("--test", action="store_true",
                        help="每个实验组合仅测试第一条基础提示词，结果独立保存至multi_fragment/test_run")
    parser.add_argument("--test-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--experiment-script", default=str(EXPERIMENT_SCRIPT), help=argparse.SUPPRESS)
    parser.add_argument("--results-dir", help=argparse.SUPPRESS)
    parser.add_argument("--task", choices=list(TASK_MODELS), help=argparse.SUPPRESS)
    parser.add_argument("--target-model", help=argparse.SUPPRESS)
    parser.add_argument("--candidate-type", choices=["mechanism", "effect"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"cuda(?::\d+)?", args.device):
        parser.error("--device 必须为 cuda 或 cuda:N")
    if args.test_worker:
        if not all([args.task, args.target_model, args.candidate_type, args.results_dir]):
            parser.error("测试子进程缺少必要参数")
        return run_single_prompt_test(args)
    try:
        list(experiment_jobs(args.tasks, model=args.model, mode=args.mode))
    except ValueError as exc:
        parser.error(str(exc))
    log_dir = RESULTS_DIR / "batch_logs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    try:
        return run_batch(args.python, args.device, Path(args.path).resolve(), log_dir,
                         dry_run=args.dry_run, tasks=args.tasks, results_dir=RESULTS_DIR,
                         cv_python=args.cv_python, nsfw_python=args.nsfw_python, model=args.model,
                         mode=args.mode, test=args.test)
    except KeyboardInterrupt:
        print("\n用户中断，停止批量运行。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
