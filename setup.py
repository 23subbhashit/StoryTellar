import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="storyscience",
    version="1.3.1",
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
        'numpy>=1.16.6',
        'pandas>=1.0.5',
        'seaborn>=0.10.1',
        'matplotlib>=3.2.2',
        'plotly>=4.9.0',
        'scikit-learn>=0.23.2',
        'tabulate>=0.8.9',
        'regex>=2.2.1',
        'nltk>=3.4.5'
    ]
)