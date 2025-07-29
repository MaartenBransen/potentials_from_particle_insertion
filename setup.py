from setuptools import setup, find_packages

setup(
    name="potentials_from_particle_insertion",
    version="1.0.1",
    author="Maarten Bransen",
    license='MIT License',
    long_description=open('README.md').read(),
    packages=find_packages(include=["potentials_from_particle_insertion", "potentials_from_particle_insertion.*"]),
    install_requires=[
        "numpy>=1.19.2",
        "scipy>=1.6.0",
    ],
)
