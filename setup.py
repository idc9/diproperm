from setuptools import setup, find_packages

# def readme():
#     with open('README.rst') as f:
#            return f.read()

install_requires = ['numpy', 'sklearn', 'statsmodels']

setup(name='diproperm',
      version='0.0.1',
      description='Implements DiProPerm for high dimensional hypothesis testing.',
      author='Iain Carmichael',
      author_email='idc9@cornell.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      test_suite='nose.collector',
      tests_require=['nose'],
      url='https://github.com/idc9/diproperm',
      download_url = 'https://github.com/idc9/diproperm/tarball/0.0.1',
      zip_safe=False)
