from setuptools import find_namespace_packages, setup

setup(
    name="s5",
    version="0.0.1",
    description="s5 in dm-haiku",
    install_requires=["dm_haiku>=0.0.7", "jax>=0.3", "numpy>=1.22", "scipy>=1.9"],
    pakcage=find_namespace_packages(),
)
