from setuptools import setup

setup(
    name='easyELM',
    version='1.0',
    description=('Extreme Learning Machine implentation'),
    author='Álvaro García González',
    author_email='alvarogrgz@gmail.com',
    url='https://github.com/alvarogrgz/easyELM',
    license='MIT',
    packages=['easyELM'],
    install_requires=['numpy', 'scipy', 'sklearn'],
    )