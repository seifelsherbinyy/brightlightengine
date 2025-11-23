from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


def _read_requirements() -> list[str]:
    requirements_path = Path(__file__).with_name("requirements.txt")
    if not requirements_path.exists():
        return []
    return [line.strip() for line in requirements_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


setup(
    name="brightstar",
    version="0.6.0",
    description="Brightstar analytics pipeline for Amazon Vendor exports",
    author="Brightstar",
    python_requires=">=3.12",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=_read_requirements(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "brightstar-pipeline=brightstar.brightstar_pipeline:main",
            "brightstar-run12=brightstar.pipeline.run_sessions12:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
