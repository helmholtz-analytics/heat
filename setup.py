import setuptools
from distutils.core import setup

setup(
    name='heat',
    packages=['heat'],
    version='0.0.0',
    description='A framework for high performace data analytics and machine learning.',
    author='Helmholtz Association',
    author_email='martin.siggel@dlr.de',
    url='https://github.com/helmholtz-analytics/heat',
    #  download_url = 'https://github.com/helmholtz-analytics/heat/archive/0.1.tar.gz', # TBD
    keywords=['data', 'analytics', 'tensors', 'distributed', 'gpu'],
    classifiers=[],
    install_requires=[
        'torch'
    ]
)
