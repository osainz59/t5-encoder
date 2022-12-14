import setuptools
import t5_encoder

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="t5_encoder",
    author="Oscar Sainz",
    version=t5_encoder.__version__,
    author_email="oscar.sainz@ehu.eus",
    description="A extension of Transformers library to include T5ForSequenceClassification class.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/osainz59/t5-encoder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["transformers", "torch"],
)