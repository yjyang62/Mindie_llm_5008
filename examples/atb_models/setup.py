# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import subprocess
import logging
import shutil
from pathlib import Path
from setuptools.command.build_py import build_py as _build_py
from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

logging.basicConfig(level=logging.INFO)
os.environ["SOURCE_DATE_EPOCH"] = "315532800" # 315532800 means 1980-01-01 00:00:00 UTC


class CustomBuildPy(_build_py):
    def run(self):
        project_root = Path(__file__).resolve().parent
        build_dir = project_root / "scripts"

        logging.info(">>> Running build.sh to compile shared libraries...")
        subprocess.check_call(["bash", "build.sh"], cwd=str(build_dir))
        
        build_pkg = Path(self.build_lib) / "atb_llm"
        (build_pkg / "lib").mkdir(parents=True, exist_ok=True)
        shutil.copytree("output/atb_models/lib", build_pkg / "lib", dirs_exist_ok=True)

        (build_pkg / "conf").mkdir(parents=True, exist_ok=True)
        shutil.copytree("atb_llm/conf", build_pkg / "conf", dirs_exist_ok=True)
        extra_dirs = ["docs", "requirements", "tools", "examples", "tests"]
        for d in extra_dirs:
            src = project_root / d
            if src.exists():
                dst = build_pkg / d
                logging.info(f"Copying {src} -> {dst}")
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                logging.warning(f"Source directory {src} does not exist, skipping.")
        super().run()


class BDistWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False


def get_version() -> str:
    """
    Return version string.

    Priority:
    1. Environment variable MINDIE_LLM_VERSION_OVERRIDE
    2. Default version
    """
    version = os.getenv("MINDIE_LLM_VERSION_OVERRIDE", "1.0.0")
    logging.info(f"Use mindie llm version: {version}")
    return version


def use_cxx11_abi() -> str:
    """
    Return whether to use CXX11 ABI as a string ("0" or "1").
    Uses torch.compiled_with_cxx11_abi() if available.
    Defaults to "0" if torch is not found or detection fails.
    """
    try:
        import torch  # noqa: F401
        abi = torch.compiled_with_cxx11_abi()
        if abi is not None:
            abi = str(int(bool(abi)))
            logging.info(f"Detect ABI from torch, set USE_CXX11_ABI to {abi}")
            return abi
    except Exception:
        logging.warning("No torch detected on current environment.")
    return "0"

setup(
    name="atb_llm",
    version=get_version(),
    author="",
    author_email="",
    description="Atb models Project",
    long_description="",
    install_requires=[],
    zip_safe=False,
    python_requires=">=3.10",
    include_package_data=False,
    packages=find_packages(),
    package_data={
        "atb_llm": [
            "lib/*"
        ]
    },
    cmdclass={
        "build_py": CustomBuildPy,
        "bdist_wheel": BDistWheel
    },
    has_ext_modules=lambda: True
)
