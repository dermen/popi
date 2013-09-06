from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

corr = Extension(
        'thor.corr',
         sources=['src/corr/correlate.pyx', 'src/corr/corr.cpp'],
         extra_compile_args={'gcc': ['--fast-math', '-O3', '-fPIC', '-Wall'],
                             'g++': ['--fast-math', '-O3', '-fPIC', '-Wall']},
         runtime_library_dirs=['/usr/lib', '/usr/local/lib'],
         extra_link_args = ['-lstdc++', '-lm'],
         include_dirs = [numpy_include, 'src/corr'],
         language='c++')

popi = Extension('thor.popi',
                 sources=['src/popi/polar_pilatus.pyx', 'src/popi/popi.cpp'],
                 extra_compile_args={'gcc': ['--fast-math', '-O3', '-fPIC', '-Wall'],
                                     'g++': ['--fast-math', '-O3', '-fPIC', '-Wall']},
                 runtime_library_dirs=['/usr/lib', '/usr/local/lib'],
                 extra_link_args = ['-lstdc++', '-lm'],
                 include_dirs = [numpy_include, 'src/popi'],
                 language='c++')

def compiler_(self):
    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        postargs = extra_postargs['gcc']
        super(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile

class custom_build_ext(build_ext):
    def build_extensions(self):
        compiler_(self.compiler)
        build_ext.build_extensions(self)

setup(  name        ='thor',
        version     ='1.0',
        cmdclass   = {'build_ext': custom_build_ext },
        ext_modules = [corr,popi],
        package_dir = {'thor': 'src'},
        packages    = ['thor']  )



