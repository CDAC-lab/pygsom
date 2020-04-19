import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pygsom",
    version="0.0.1",
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
    python_requires='>=3.6',
)