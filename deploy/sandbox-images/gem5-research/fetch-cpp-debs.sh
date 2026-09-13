#!/bin/sh
# Fetch the packages Dockerfile.add-cpp installs, on the HOST.
#
# Why not `apt-get install g++` inside the build: on this network
# deb.debian.org resolves to 198.18.0.71 -- RFC 2544 benchmarking space, the
# synthetic address a VPN or intercept layer hands out -- and apt inside the VM
# ran at 1416 B/s, fetching 581 kB in 6m51s before the 9 MB Packages index
# failed outright with "Connection failed". The host reaches the same mirror at
# ~62 kB/s: poor, but forty times better and enough. So the download happens
# here and the build stays offline, which the sandbox is anyway at run time.
#
# The versions are pinned to what the image already carries. g++-12 must match
# the installed gcc-12 exactly or dpkg refuses it, which makes this more
# reproducible than the `apt-get install g++` it replaces, not less.
set -eu

GCC12_VERSION=12.2.0-14+deb12u1   # must equal the image's gcc-12 and libgcc-12-dev
GCC_DEFAULTS_VERSION=12.2.0-3     # provides the /usr/bin/g++ the tool invokes
ARCH=arm64
BASE=http://deb.debian.org/debian/pool/main/g

cd "$(dirname "$0")"

fetch() {
    if [ -f "$2" ]; then echo "have $2"; return; fi
    echo "fetching $2"
    curl -fSL --retry 5 --retry-all-errors --connect-timeout 20 -o "$2.part" "$1"
    mv "$2.part" "$2"
}

fetch "$BASE/gcc-12/g++-12_${GCC12_VERSION}_${ARCH}.deb" "g++-12_${GCC12_VERSION}_${ARCH}.deb"
fetch "$BASE/gcc-12/libstdc++-12-dev_${GCC12_VERSION}_${ARCH}.deb" "libstdc++-12-dev_${GCC12_VERSION}_${ARCH}.deb"
fetch "$BASE/gcc-defaults/g++_${GCC_DEFAULTS_VERSION}_${ARCH}.deb" "g++_${GCC_DEFAULTS_VERSION}_${ARCH}.deb"

echo "done:"
ls -la ./*.deb
