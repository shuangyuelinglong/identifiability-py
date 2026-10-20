# -*- coding: utf-8 -*-
"""
设置 GitHub 用户名与仓库名（替换 pyproject.toml / CITATION.cff / README.md 里的链接）。

用法：
    python tools/set_repo.py <username> <repo-name>
    例：python tools/set_repo.py shuangyuelinglong identifiability-py
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["pyproject.toml", "CITATION.cff", "README.md"]
PAT = re.compile(r"https://github\.com/[^/\s\"\)]+/[A-Za-z0-9._-]+")

USER_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
REPO_RE = re.compile(r"[A-Za-z0-9._-]{1,100}")


def main():
    if len(sys.argv) != 3:
        print("用法: python tools/set_repo.py <github-username> <repo-name>")
        return 1
    user, repo = sys.argv[1].strip(), sys.argv[2].strip()
    if not USER_RE.fullmatch(user):
        print(f"  ** 用户名 '{user}' 不合法：只能含英文字母、数字、连字符")
        return 1
    if not REPO_RE.fullmatch(repo):
        print(f"  ** 仓库名 '{repo}' 不合法：只能含字母、数字、点、下划线、连字符")
        return 1
    new = f"https://github.com/{user}/{repo}"
    changed = 0
    for f in FILES:
        p = ROOT / f
        t = p.read_text(encoding="utf-8")
        t2, k = PAT.subn(new, t)
        if k:
            p.write_text(t2, encoding="utf-8")
            changed += k
            print(f"  {f}: 替换 {k} 处")
    if changed == 0:
        print("  没有找到可替换的 URL")
    print()
    for f in FILES:
        for m in PAT.finditer((ROOT / f).read_text(encoding="utf-8")):
            print(f"  {f}: {m.group(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
