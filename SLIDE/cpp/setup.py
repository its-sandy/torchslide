from setuptools import setup, Extension
from torch.utils import cpp_extension

cpp_module = cpp_extension.CppExtension('cppslide_cpp',
                                        sources=['cppslide.cpp'],
                                        extra_compile_args=['-fopenmp'],
                                        extra_link_args=['-lgomp']
                                        )

setup(name='cppslide_cpp',
      ext_modules=[cpp_module],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

cpp_module = cpp_extension.CppExtension('srpHash_cpp',
                                        sources=['srpHash.cpp'],
                                        extra_compile_args=['-fopenmp'],
                                        extra_link_args=['-lgomp']
                                        )

setup(name='srpHash_cpp',
      ext_modules=[cpp_module],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
