from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'squirtle_chase_object'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='snaseem8',
    maintainer_email='snaseem8@gatech.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['find_object=squirtle_chase_object.find_object:main',
			'get_object_range=squirtle_chase_object.get_object_range:main',
            'chase_object=squirtle_chase_object.chase_object:main',
            'debug=squirtle_chase_object.debug:main'
        ],
    },
)
