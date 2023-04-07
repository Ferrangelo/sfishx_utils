from setuptools import find_packages, setup

setup(
    name="sfishx_utils",
    version="0.0",
    description="utils to read and plot sfx fisher matrices",
    zip_safe=False,
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19",
        "scipy",
    ],
)
