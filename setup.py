from setuptools import setup

setup(
   name='DL_utils',
   version='0.1',
   description='Deep Learning utils for Keras over Tensorflow',
   author='madisi98',
   author_email='madisi1998@gmail.com',
   packages=['DL_utils'],  #same as name
   install_requires=['numpy', 'tensorflow'], #external packages as dependencies
)
