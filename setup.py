from setuptools import setup

setup(name='dnacycp',
      packages=['dnacycp'],
      version='0.0.1dev1',
      python_requires='>3.7.0',
      install_requires=[
      'numpy==1.21.5',
      'pandas==1.3.5',
      'tensorflow==2.7.0',
      'keras==2.7.0',
      'bio==1.3.3',
      'docopt==0.6.2'
      ],
      entry_points={
            'console_scripts': ['dnacycp-cli=dnacycp.cli:main']
      }
      )
