Name:		tensor-filter-swift_yolo
Summary:	NNStreamer tensor-filter subplugin for swift_yolo
Version:	1.0.0
Release:	0
Group:		Development/Libraries
Packager:	MyungJoo Ham <myungjoo.ham@samsung.com>
# You may change the License to anything you want (Proprietary is allowed)
License:	Proprietary
Source0:	tensor-filter-swift_yolo-%{version}.tar.gz
Source1001:	tensor-filter-swift_yolo.manifest

Requires:	nnstreamer
BuildRequires:	nnstreamer-devel
BuildRequires:	meson
BuildRequires:	glib2-devel
BuildRequires:	gstreamer-devel

%description
Fill this in!

%prep
%setup -q
cp %{SOURCE1001} .

%build
mkdir -p build
meson --prefix=%{_prefix} build
ninja -C build %{?_smp_mflags}

%install
DESTDIR=%{buildroot} ninja -C build %{?_smp_mflags} install

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%files
%manifest tensor-filter-swift_yolo.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/*.so

%changelog
* Fri Oct 11 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
- Initial Template Tensor-Filter of 1.0.0
