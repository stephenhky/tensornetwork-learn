
from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


def install_requirements():
    return [package_string.strip() for package_string in open('requirements.txt', 'r')]


def package_description():
    text = open('README.md', 'r').read()
    startpos = text.find('## Introduction')
    return text[startpos:]


setup(name='tensornetwork-learn',
      version="0.0.1",
      description="Machine Learning using Tensor Networks",
      long_description=package_description(),
      long_description_content_type='text/markdown',
      classifiers=[
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Topic :: Software Development :: Version Control :: Git",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "License :: OSI Approved :: MIT License",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers"
      ],
      keywords="machine learning, tensornetwork",
      url="https://github.com/stephenhky/tensornetwork-learn",
      author="Kwan-Yuet Ho",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      packages=['tensorml'],
      install_requires=install_requirements(),
      tests_require=[
          'unittest2',
      ],
      # scripts=[],
      include_package_data=True,
      test_suite="test",
      zip_safe=False)
