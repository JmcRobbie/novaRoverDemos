from setuptools import setup, find_packages

setup(name='nova_rover_demos',
      version='0.1',
      description='Proof-of-concept demonstrations for the Nova Rover autonomous system',
      url='https://github.com/JmcRobbie/novaRoverDemos',
      author='Jack McRobbie',
      author_email='jmcrobbie@melbournespace.com.au',
      license='MIT',
      packages = find_packages(),
      zip_safe=False
)