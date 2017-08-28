try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='turbulence_postprocessing',
      version='0.1',
      description='Turbulence post-processing scripts',
      long_description=readme(),
      keywords='turbulence post processing, computational fluid dynamics',
      url='https://github.com/marchdf/turbulence_postprocessing',
      download_url='https://github.com/marchdf/turbulence_postprocessing',
      author='Marc T. Henry de Frahan',
      author_email='marchdf@gmail.com',
      license='Apache License 2.0',
      packages=['turbulence_postprocessing'],
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'sphinx_rtd_theme'
      ],
      test_suite='tests',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
