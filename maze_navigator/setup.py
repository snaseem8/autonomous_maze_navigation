from setuptools import find_packages, setup

package_name = 'maze_navigator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/maze_navigator/launch', ['launch/maze_navigator_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='snaseem8',
    maintainer_email='snaseem8@gatech.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['sign_classifier_node=maze_navigator.sign_classifier_node:main',
			'navigator_node=maze_navigator.navigator_node:main',
            'pose_listener=maze_navigator.pose_listener:main',
            'debug=maze_navigator.debug:main'
        ],
    },
)
