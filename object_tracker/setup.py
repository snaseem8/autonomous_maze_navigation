from setuptools import find_packages, setup

package_name = 'squirtle_object_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='snaseem8',
    maintainer_email='snaseem8@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['find_object=squirtle_object_follower.find_object:main',
			'rotate_robot=squirtle_object_follower.rotate_robot:main',
            'debug=squirtle_object_follower.debug:main'
        ],
    },
)
