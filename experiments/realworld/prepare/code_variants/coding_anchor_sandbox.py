#!/usr/bin/env python3
"""Persistent test controller inside a private user/mount/PID/network namespace.

The launcher exposes only read-only system Python libraries and this controller.
Each candidate runs in a fresh child, without capabilities or process/network
syscalls, with CPU/address-space/file limits. No home, shared filesystem, credentials, GPU,
or host processes are mounted. This is a data-verification tool, not training.
"""
import ast
import ctypes
import errno
import json
import os
import resource
import signal
import sys
import time


def restrict_child():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    libc = ctypes.CDLL(None, use_errno=True)
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
    class Caps(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                    ("inheritable", ctypes.c_uint32)]
    header, caps = Header(0x20080522, 0), (Caps * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(caps)) != 0:
        raise RuntimeError("capset failed")
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise RuntimeError("no_new_privs failed")
    sc = ctypes.CDLL("libseccomp.so.2")
    sc.seccomp_init.argtypes = [ctypes.c_uint32]
    sc.seccomp_init.restype = ctypes.c_void_p
    sc.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sc.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sc.seccomp_load.argtypes = [ctypes.c_void_p]
    sc.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = sc.seccomp_init(0x7fff0000)  # ALLOW, with explicit dangerous calls denied.
    if not ctx:
        raise RuntimeError("seccomp init failed")
    for name in ("clone", "clone3", "fork", "vfork", "execve", "execveat", "ptrace",
                 "kill", "tkill", "tgkill", "process_vm_writev", "process_vm_readv",
                 "mount", "umount2", "pivot_root", "chroot", "unshare", "setns",
                 "socket", "socketpair", "connect", "bind", "listen", "accept",
                 "bpf", "perf_event_open", "open_by_handle_at", "io_uring_setup",
                 "reboot", "kexec_load", "init_module", "finit_module"):
        nr = sc.seccomp_syscall_resolve_name(name.encode())
        if nr >= 0 and sc.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, nr, 0) != 0:
            raise RuntimeError("seccomp rule failed")
    if sc.seccomp_load(ctx) != 0:
        raise RuntimeError("seccomp load failed")
    sc.seccomp_release(ctx)


def check(job):
    try:
        solution = compile(job["code"], "<solution>", "exec")
        tests = []
        for text in job["tests"]:
            tree = ast.parse(text.strip())
            if not any(isinstance(stmt, ast.Assert) for stmt in tree.body):
                return {"status": "unsupported_test_shape"}
            tests.append(compile(tree, "<test>", "exec"))
        if not tests:
            return {"status": "no_tests"}
    except (SyntaxError, ValueError, TypeError):
        return {"status": "invalid_test_syntax"}
    started = time.monotonic()
    child = os.fork()
    if child == 0:
        try:
            null = os.open("/dev/null", os.O_RDWR)
            for fd in (0, 1, 2):
                os.dup2(null, fd)
            os.closerange(3, 256)
            restrict_child()
            for test in tests:
                namespace = {"__name__": "__main__"}
                exec(solution, namespace)
                exec(test, namespace)
            os._exit(71)
        except BaseException:
            os._exit(72)
    timeout = False
    while True:
        pid, status = os.waitpid(child, os.WNOHANG)
        if pid:
            break
        if time.monotonic() - started > 3:
            os.kill(child, signal.SIGKILL)
            _, status = os.waitpid(child, 0)
            timeout = True
            break
        time.sleep(0.002)
    return {"status": "pass" if os.waitstatus_to_exitcode(status) == 71 else
            ("timeout" if timeout else "test_or_runtime_failure"),
            "n_tests": len(tests), "seconds": round(time.monotonic() - started, 6)}


if __name__ == "__main__":
    for line in sys.stdin:
        print(json.dumps(check(json.loads(line))), flush=True)
