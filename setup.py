import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pygsom",
    version="0.1.0",
    author="Thimal Kempitiya",
    author_email="t.kempitiya@gmail.com",
    description="gsom clustering and dimensional reduction algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/thimalk/pygsom/src",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
            'numpy>=1.17.4',
            'pandas>=0.25.3',
            'scipy>=1.4.1',
            'tqdm>=4.39.0',
            'matplotlib>=3.1.2',
    ],
    python_requires='>=3.6',
)