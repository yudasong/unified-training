#!/bin/bash
# Start the test controller in private user/mount/net/pid namespaces with a read-only system /usr, /lib, /lib64.
#   coding_anchor_sandbox.sh <private empty dir> <absolute controller script> [system python, default python3.11]
set -euo pipefail
root="${1:?private empty scratch directory}"
runner="${2:?absolute controller script}"
python_bin="${3:-/usr/bin/python3.11}"
exec unshare --user --map-root-user --mount --net --pid --fork --kill-child \
  /bin/bash -ceu '
    root="$1"; runner="$2"; python_bin="$3"
    mount --make-rprivate /
    mount -t tmpfs -o size=64m,nosuid,nodev tmpfs "$root"
    mkdir -p "$root"/{usr,lib,lib64,tmp,proc,dev}
    for dir in usr lib lib64; do
      mount --bind "/$dir" "$root/$dir"
      mount -o remount,bind,ro,nosuid,nodev "$root/$dir"
    done
    ln -s usr/bin "$root/bin"
    touch "$root/dev/null" "$root/runner.py"
    mount --bind /dev/null "$root/dev/null"
    mount --bind "$runner" "$root/runner.py"
    mount -o remount,bind,ro,nosuid,nodev "$root/runner.py"
    mount -t proc -o nosuid,nodev,noexec proc "$root/proc"
    cd "$root"
    exec chroot "$root" "$python_bin" -I /runner.py
  ' sandbox "$root" "$runner" "$python_bin"
