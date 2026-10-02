from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


setup(
    name="pic_kde_2d",
    version="0.1.0",
    packages=find_packages(),
    install_requires=(ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines(),
    python_requires=">=3.10",
)
