import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="storyscience",
    version="0.0.1",
    author="subbhashit mukherjee",
    author_email="subbhashitmukherjee@gmail.com",
    packages=["storyscience"],
    description="A storyteller",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/23subbhashit/StoryTellar/",
    license='MIT',
    python_requires='>=3.6',
    install_requires=[
    
    ]
)