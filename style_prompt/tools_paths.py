"""模型目录相对于本文件定位，迁移时保留 style_prompt/ 与 tools/ 同级。"""

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
TOOLS_ROOT = Path(os.environ.get("STYLEPROMPT_TOOLS_ROOT", PACKAGE_DIR.parent / "tools")).expanduser()


def resolve_tools_asset_path(value):
    """解析相对于 tools/ 的路径，并兼容旧配置中的 ../tools/ 或绝对路径。

    旧主机的 Hugging Face snapshot 路径不存在时，按仓库和版本重新下载
    到当前主机的系统缓存；tools/ 内的资产仍直接从项目目录读取。
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        # 旧 runtime_assets.json 可能包含另一台主机的 tools 绝对路径。
        if "tools" in path.parts:
            index = path.parts.index("tools")
            return TOOLS_ROOT.joinpath(*path.parts[index + 1:])
        if not path.exists():
            for index, part in enumerate(path.parts):
                if (part.startswith("models--")
                        and len(path.parts) > index + 2
                        and path.parts[index + 1] == "snapshots"):
                    from huggingface_hub import snapshot_download

                    repo_id = part[len("models--"):].replace("--", "/")
                    revision = path.parts[index + 2]
                    snapshot = Path(snapshot_download(repo_id=repo_id, revision=revision))
                    return snapshot.joinpath(*path.parts[index + 3:])
        return path
    if path.parts[:2] == ("..", "tools"):
        return TOOLS_ROOT.joinpath(*path.parts[2:])
    if path.parts[:1] == ("tools",):
        return TOOLS_ROOT.joinpath(*path.parts[1:])
    return TOOLS_ROOT / path
