from setuptools import find_packages, setup
import ranky

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
     name='ranky',
     version=ranky.__version__,
     author="Adrien Pavao",
     author_email="adrien.pavao@gmail.com",
     description="Compute rankings in Python.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/didayolo/ranky",
     project_urls={
            "Bug Tracker": "https://github.com/didayolo/ranky/issues",
            "Documentation": "https://didayolo.github.io/ranky/",
            "Source Code": "https://github.com/didayolo/ranky",
     },
     packages=find_packages(),
     include_package_data=True,
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: Unix",
     ],
     install_requires=[
        'pandas', 'numpy', 'matplotlib', 'scikit-learn>=0.23.2', 
        'scipy', 'seaborn', 'networkx', 'python-Levenshtein', 'tqdm', 'baycomp',
     ],
 )
