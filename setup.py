from setuptools import find_namespace_packages, setup

setup(
    name="s5",
    version="0.0.1",
    description="s5 in dm-haiku",
    requires=["dm-haiku>=0.0.7", "jax>=0.3"],
    pakcage=find_namespace_packages(),
)
