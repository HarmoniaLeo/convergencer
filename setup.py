from setuptools import setup
 
setup(name='convergencer',
      version="0.1.8",
      description='Data science workflow of automatic data reading, preprocessing, model selection, model integration, model parameter adjustment and effect evaluation.',
      author='HarmoniaLeo',
      author_email='1532282692@qq.com',
      maintainer='HarmoniaLeo',
      maintainer_email='1532282692@qq.com',
      packages=['convergencer.data','convergencer.models','convergencer.utils','convergencer.processors'],
      license="Public domain",
      platforms=["any"],
     )