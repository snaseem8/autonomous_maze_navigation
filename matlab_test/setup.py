from setuptools import find_packages, setup

package_name = 'matlab_test'

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
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['simple_pub=matlab_test.simple_pub:main', 
                            'simple_client= matlab_test.simple_client:main',
                            'simple_service= matlab_test.simple_service:main'
        ],
    },
)
