from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description="This model predicts a movie's gross revenue based on select feature data and generates movie recommendations based on user input",
    author='Lily Ren',
    license='MIT',
    install_requires=['pandas','wordcloud','nltk','cpi','matplotlib']
)
