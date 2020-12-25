from setuptools import setup, Extension
from torch.utils import cpp_extension

cpp_module = cpp_extension.CppExtension('cppSparseMultiply_cpp',
                                        sources=['cppSparseMultiply.cpp'],
                                        extra_compile_args=['-fopenmp'],
                                        extra_link_args=['-lgomp']
                                        )

setup(name='cppSparseMultiply_cpp',
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

cpp_module = cpp_extension.CppExtension('bucketsTable_cpp',
                                        sources=['bucketsTable.cpp'],
                                        extra_compile_args=['-fopenmp'],
                                        extra_link_args=['-lgomp']
                                        )

setup(name='bucketsTable_cpp',
      ext_modules=[cpp_module],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
