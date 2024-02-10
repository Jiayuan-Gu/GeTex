from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="getex",
    author="Jiayuan Gu",
    description="Get Textures for CAD from Images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"getex": "src"},
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
)
