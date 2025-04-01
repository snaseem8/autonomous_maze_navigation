from setuptools import find_packages, setup

package_name = 'squirtle_waypoint_nav'

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
    maintainer_email='snaseem8@gatech.edu',
    description='Lab5 Waypoint Publisher',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['publish_waypoints=squirtle_waypoint_nav.publish_waypoints:main'
        ],
    },
)
