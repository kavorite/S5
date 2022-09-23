from setuptools import find_namespace_packages, setup

setup(
    name="s5",
    version="0.0.1",
    description="s5 in dm-haiku",
    install_requires=["dm_haiku", "jax", "numpy", "scipy"],
    package=find_namespace_packages("_src"),
)
