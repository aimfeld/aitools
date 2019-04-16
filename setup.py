import setuptools

setuptools.setup(
    name='aitools-aimfeld',
    version='0.3.3',
    description='Python tools for datascience and AI',
    url='https://github.com/aimfeld/aitools',
    author='Adrian Imfeld',
    author_email='aimfeld@aimfeld.ch',
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False
)

# Update package
# python setup.py sdist bdist_wheel
# python -m twine upload dist/*
