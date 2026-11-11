from pathlib import Path
from setuptools import find_packages, setup

root = Path(__file__).parent
subpackages = find_packages("python")
packages = ["pygo"] + ["pygo." + p for p in subpackages]
setup(
    name="pygo",
    version=(root / "VERSION").read_text().strip(),
    description="PyGo production tensor and deep-learning runtime",
    long_description=(root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"pygo": "python"},
    packages=packages,
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=["numpy>=1.24"],
)
