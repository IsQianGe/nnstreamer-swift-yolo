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

# cmake_project = cxx.find_program('cmake')

# # 定义 ExternalProject
# cmake_project_dep = declare_dependency(
#     sources: 'components/ncnn/',
#     include_directories: 'components/ncnn/include',
#     link_with: [],
#     link_args: [],
#     dependencies: []
# )

# # 配置 ExternalProject
# configure_command = [
#     cmake_project,
#     '-DCMAKE_INSTALL_PREFIX=' + cxx.get_builtin_option('prefix'),
#     '-DCMAKE_BUILD_TYPE=' + cxx.get_builtin_option('buildtype'),
#     '-G', 'Unix Makefiles',
#     meson.source_root(),
# ]

# build_command = [
#     cmake_project,
#     '--build', '.',
#     '--config', meson.get_compiler('cpp').get_builtin_option('buildtype'),
# ]

library_dep=cxx.find_library('ncnn', dirs : join_paths(meson.source_root(),'components/ncnn/build-host-gcc-linux/install/lib'))
library_include_dir = join_paths(meson.source_root(),'components/ncnn/build-host-gcc-linux/install/include/ncnn')

subplugin_shared = shared_library('nnstreamer_filter_swift_yolo',
  sources,
  dependencies: [base_deps, library_dep],
  include_directories : library_include_dir,
  install: true,
  install_dir: filter_subplugin_install_dir,
)
