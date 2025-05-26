from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import include_paths

setup(
    name='pathweaver',
    ext_modules=[

# CPU sign bit generation
        CUDAExtension(
            name='cpu_generate_sign_bit',
            sources=['./csrc/cpu_generate_sign_bit.cpp'],
            include_dirs=include_paths(),
            extra_compile_args={
                'cxx': ['-O3', '-fopenmp']
            }
        ),

# GPU search all together without gpu copy
        CUDAExtension(
            name='pathweaver',
            sources=['./csrc/pathweaver.cu'],
            include_dirs=include_paths(),
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-Xptxas=-v', '-arch=sm_80']
            }
        ),

#=====================================================================================================

    ],
    
    cmdclass={'build_ext': BuildExtension}
)