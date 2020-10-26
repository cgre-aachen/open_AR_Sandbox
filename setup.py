from setuptools import setup, find_packages
version = '0.1'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='arsandbox',
    version=version,
    packages=find_packages(exclude=('test', 'docs')),
    include_package_data=True,
    install_requires=[
        'matplotlib >= 3.2.1',
        'numpy',
        'pandas',
        'panel',
        'scipy',
        'scikit-image',
        'opencv-contrib-python',
        'pytest',
        'jupyter',
        'seaborn'
    ],
    url='https://github.com/cgre-aachen/open_AR_Sandbox',
    license='LGPL v3',
    author='Simon Virgo, Daniel Escallon, Miguel de la Varga',
    author_email='simon@terranigma-solutions.com',
    description='An Open-source, Python-based Augmented reality (AR) sandbox to display many modules.',
    keywords=['AR', 'Augmented reality', 'sandbox', 'geology']
)
