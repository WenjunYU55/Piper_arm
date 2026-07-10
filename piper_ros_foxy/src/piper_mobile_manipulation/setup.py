from setuptools import find_packages, setup

package_name = 'piper_mobile_manipulation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Safe mobile manipulation stack for PiPER arm perception and fake commands.',
    license='Apache-2.0',
    tests_require=['pytest'],
)
