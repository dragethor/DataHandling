from setuptools import find_packages, setup

setup(
    name='DataHandling',
    packages = find_packages(),
    package_dir = {"":"src"},
    version='0.1.0',
    description='Master Project combining DNS and machine learning',
    author='Thor Christensen',
    license='',
)

#    packages = find_packages(
#        where = 'src',
#        exclude = ['data','notebooks','references','reports','*.ipynb',]
#    ),
#    package_dir = {"":"src"},