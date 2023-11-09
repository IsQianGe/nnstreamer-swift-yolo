project('nnstreamer', 'c', 'cpp',
  version: '1.0.0',
  license: ['Proprietary'],
  meson_version: '>=0.50.0',
)

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

path_prefix = get_option('prefix')
subplugin_install_prefix = join_paths(path_prefix, 'lib', 'nnstreamer')
filter_subplugin_install_dir = join_paths(subplugin_install_prefix, 'filters')
log = join_paths(subplugin_install_prefix, 'log')

glib_dep = dependency('glib-2.0')
gmodule_dep = dependency('gmodule-2.0')
gst_dep = dependency('gstreamer-1.0')
nnstreamer_dep = dependency('nnstreamer')

base_deps = [
  glib_dep,
  gmodule_dep,
  gst_dep,
  nnstreamer_dep
]

sources = [
  'src/tensor_filter_swift_yolo.cc'
]

library_dep=cxx.find_library('ncnn', dirs : join_paths(meson.source_root(),'components/ncnn/build-aarch64-linux-gnu/install/lib'))
library_include_dir = 'components/ncnn/build-aarch64-linux-gnu/install/include/ncnn'

subplugin_shared = shared_library('nnstreamer_filter_swift_yolo',
  sources,
  dependencies: [base_deps, library_dep],
  include_directories : library_include_dir,
  install: true,
  install_dir: filter_subplugin_install_dir,
  c_args: ['-fpermissive'],
  cpp_args: ['-fpermissive','-fopenmp']
)