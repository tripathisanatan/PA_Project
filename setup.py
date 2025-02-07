from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='PA-project',
    version='0.0.1',
    author='Sanatan',
    author_email='tripathisanatan3@gmail.com',
    packages=find_packages(),  # This will now detect packages from the root, not 'src'
    package_dir={'PA_project': 'src'},  # Maps package 'PA_project' to 'src'
    install_requires=get_requirements('requirements.txt'),
    include_package_data=True
)
