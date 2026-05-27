import ast
import os
import re
import shutil
import setuptools
import subprocess
import sys
import torch
import platform
import urllib
import urllib.error
import urllib.request
from setuptools import find_packages
from setuptools.command.build_py import build_py
from packaging.version import parse
from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from scripts.generate_pyi import generate_pyi_file

source_package_name = 'deep_gemm'
wheel_name = 'deep_gemm_moe_L1'
package_name = 'deep_gemm_moe_L1'

DG_SKIP_CUDA_BUILD = int(os.getenv('DG_SKIP_CUDA_BUILD', '0')) == 1
DG_FORCE_BUILD = int(os.getenv('DG_FORCE_BUILD', '0')) == 1
DG_USE_LOCAL_VERSION = int(os.getenv('DG_USE_LOCAL_VERSION', '1')) == 1
DG_JIT_USE_RUNTIME_API = int(os.environ.get('DG_JIT_USE_RUNTIME_API', '0')) == 1

# Compiler flags
cxx_flags = ['-std=c++17', '-O3', '-fPIC', '-Wno-psabi', '-Wno-deprecated-declarations',
             f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}']
cuda_flags = ['-std=c++17', '-O3',
              f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}']
if DG_JIT_USE_RUNTIME_API:
    cxx_flags.append('-DDG_JIT_USE_RUNTIME_API')
    cuda_flags.append('-DDG_JIT_USE_RUNTIME_API')

# Sources
current_dir = os.path.dirname(os.path.realpath(__file__))
sources = ['csrc/python_api.cpp']
build_include_dirs = [
    f'{CUDA_HOME}/include',
    f'{CUDA_HOME}/include/cccl',
    os.path.join(current_dir, source_package_name, 'include'),
    os.path.join(current_dir, 'third-party/cutlass/include'),
    os.path.join(current_dir, 'third-party/fmt/include'),
]
build_libraries = ['cudart', 'nvrtc', 'nccl']
build_library_dirs = [f'{CUDA_HOME}/lib64']
third_party_include_dirs = [
    'third-party/cutlass/include/cute',
    'third-party/cutlass/include/cutlass',
]

# Release
base_wheel_url = 'https://github.com/DeepSeek-AI/DeepGEMM/releases/download/{tag_name}/{wheel_name}'


def get_package_version():
    with open(Path(current_dir) / source_package_name / '__init__.py', 'r') as f:
        version_match = re.search(r'^__version__\s*=\s*(.*)$', f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))

    revision = ''
    if DG_USE_LOCAL_VERSION:
        # noinspection PyBroadException
        try:
            status_cmd = ['git', 'status', '--porcelain']
            status_output = subprocess.check_output(status_cmd).decode('ascii').strip()
            if status_output:
                print(f'Warning: Git working directory is not clean. Uncommitted changes:\n{status_output}')
                assert False, 'Git working directory is not clean'

            cmd = ['git', 'rev-parse', '--short', 'HEAD']
            revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
        except Exception:
            revision = '+local'
    return f'{public_version}{revision}'


def get_platform():
    if sys.platform.startswith('linux'):
        return f'linux_{platform.uname().machine}'
    else:
        raise ValueError('Unsupported platform: {}'.format(sys.platform))


def get_wheel_url():
    torch_version = parse(torch.__version__)
    torch_version = f'{torch_version.major}.{torch_version.minor}'
    python_version = f'cp{sys.version_info.major}{sys.version_info.minor}'
    platform_name = get_platform()
    deep_gemm_version = get_package_version()
    cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)

    # Determine the version numbers that will be used to determine the correct wheel
    # We're using the CUDA version used to build torch, not the one currently installed
    cuda_version = parse(torch.version.cuda)
    cuda_version = f'{cuda_version.major}'

    # Determine wheel URL based on CUDA version, torch version, python version and OS
    wheel_filename = f'{wheel_name}-{deep_gemm_version}+cu{cuda_version}-torch{torch_version}-cxx11abi{cxx11_abi}-{python_version}-{platform_name}.whl'
    wheel_url = base_wheel_url.format(tag_name=f'v{deep_gemm_version}', wheel_name=wheel_filename)
    return wheel_url, wheel_filename


def get_ext_modules():
    if DG_SKIP_CUDA_BUILD:
        return []

    return [CUDAExtension(name=f'{package_name}._C',
                          sources=sources,
                          include_dirs=build_include_dirs,
                          libraries=build_libraries,
                          library_dirs=build_library_dirs,
                          extra_compile_args={'cxx': cxx_flags, 'nvcc': cuda_flags})]


class CustomBuildPy(build_py):
    def run(self):
        # First, prepare the include directories
        self.prepare_includes()

        # Second, make clusters' cache setting default into `envs.py`
        self.generate_default_envs()

        # Third, generate and copy .pyi file to build root directory
        self.generate_pyi_file()

        # Finally, run the regular build
        build_py.run(self)

    def generate_pyi_file(self):
        generate_pyi_file(name='_C', root='./csrc', output_dir='./stubs')
        pyi_source = os.path.join(current_dir, 'stubs', '_C.pyi')
        pyi_target = os.path.join(self.build_lib, package_name, '_C.pyi')

        if os.path.exists(pyi_source):
            print(f"Copying .pyi file from {pyi_source} to {pyi_target}")
            os.makedirs(os.path.dirname(pyi_target), exist_ok=True)
            shutil.copy2(pyi_source, pyi_target)
        else:
            print(f"Warning: .pyi file not found at {pyi_source}")

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        for name in ('DG_JIT_CACHE_DIR', 'DG_JIT_PRINT_COMPILER_COMMAND', 'DG_JIT_CPP_STANDARD'):
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''

        with open(os.path.join(self.build_lib, package_name, 'envs.py'), 'w') as f:
            f.write(code)

    def prepare_includes(self):
        # Create temporary build directory instead of modifying package directory
        build_include_dir = os.path.join(self.build_lib, package_name, 'include')
        os.makedirs(build_include_dir, exist_ok=True)

        # Copy third-party includes to the build directory
        for d in third_party_include_dirs:
            dirname = d.split('/')[-1]
            src_dir = os.path.join(current_dir, d)
            dst_dir = os.path.join(build_include_dir, dirname)

            # Remove existing directory if it exists
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)

            # Copy the directory
            shutil.copytree(src_dir, dst_dir)


class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        if DG_FORCE_BUILD or DG_USE_LOCAL_VERSION:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print(f'Try to download wheel from URL: {wheel_url}')
        # noinspection PyBroadException
        try:
            with urllib.request.urlopen(wheel_url, timeout=1) as response:
                with open(wheel_filename, 'wb') as out_file:
                    data = response.read()
                    out_file.write(data)

            # Make the archive
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)
            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f'{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}'
            wheel_path = os.path.join(self.dist_dir, archive_basename + '.whl')
            os.rename(wheel_filename, wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print('Precompiled wheel not found. Building from source...')
            # If the wheel could not be downloaded, build from source
            super().run()


def get_packages():
    packages = []
    for name in find_packages('.'):
        if name == source_package_name or name.startswith(f'{source_package_name}.'):
            packages.append(package_name + name[len(source_package_name):])
    return packages


if __name__ == '__main__':
    # noinspection PyTypeChecker
    setuptools.setup(
        name=f'{wheel_name}',
        version=get_package_version(),
        packages=get_packages(),
        package_dir={package_name: source_package_name},
        package_data={
            package_name: [
                'include/deep_gemm/**/*',
                'include/cute/**/*',
                'include/cutlass/**/*',
            ]
        },
        ext_modules=get_ext_modules(),
        zip_safe=False,
        cmdclass={
            'build_py': CustomBuildPy,
            'build_ext': BuildExtension,
            'bdist_wheel': CachedWheelsCommand,
        },
    )
