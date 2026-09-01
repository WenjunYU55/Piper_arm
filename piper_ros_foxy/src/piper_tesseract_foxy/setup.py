from setuptools import find_packages, setup


package_name = 'piper_tesseract_foxy'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/tesseract_bridge_params.yaml']),
        ('share/' + package_name + '/launch', ['launch/tesseract_foxy.launch.py']),
        ('share/' + package_name + '/model', [
            'model/piper.srdf',
            'model/piper_bunker.srdf',
            'model/piper_plugins.yaml',
            'model/contact_manager_plugins.yaml',
            'model/collision_model.yaml',
            'model/collision_model_ground.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=False,
    maintainer='Piper arm maintainers',
    maintainer_email='root@todo.todo',
    description='Command-free ROS 2 Foxy adapter for isolated Tesseract planning.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'motion_planner_bridge = piper_tesseract_foxy.bridge_node:main',
            'tesseract_plan_bridge = piper_tesseract_foxy.bridge_node:main',
            'tesseract_plan_worker = piper_tesseract_foxy.worker:main',
            'piper_capability_map_generator = '
            'piper_tesseract_foxy.capability_map_generator:main',
        ],
    },
)
