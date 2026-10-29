"""In-process guard for the code-reward sandbox (installed before any test code runs).

``install(workdir)`` registers a ``sys.addaudithook`` hook that raises on
filesystem mutation outside ``workdir`` (remove, rename, mkdir, chmod, writes),
on any deletion at all outside ``workdir``, on process creation, signals,
``os.chdir``, environment changes, sockets and other network clients, ctypes,
and on attempts to add further audit hooks.  Reads are allowed everywhere so
imports keep working.  Audit hooks cannot be removed once installed, so a test
solution cannot undo this from Python; it is a defence against generated code
under RL, not against a determined attacker.

``read_nonce(workdir)`` returns the run nonce from ``.nonce`` and deletes the
file so the tested code cannot read it; the runner prints ``PASS_LINE(nonce)``
only after every test passed, and code_reward accepts a run only when that line
is present and the exit code is 0.
"""
import os
import sys

_BLOCKED_PREFIXES = (
    "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
    "os.forkpty", "os.kill", "os.killpg",
    "os.startfile", "os.add_dll_directory", "pty.", "socket.", "ftplib.", "smtplib.",
    "poplib.", "imaplib.", "nntplib.", "telnetlib.", "http.client.", "urllib.", "webbrowser.",
    "ctypes.", "sys.addaudithook", "shutil.chown", "shutil.unpack_archive", "os.setuid",
    "os.setgid", "os.setpgid", "os.setsid", "os.setpriority", "resource.setrlimit",
    "os.scandir", "os.listdir",  # limited below to the sandbox and the interpreter
)
_MUTATIONS = ("os.remove", "os.rmdir", "os.rename", "os.truncate", "os.chmod", "os.chown",
              "os.link", "os.symlink", "os.mkdir", "os.utime", "shutil.rmtree", "shutil.move",
              "shutil.copytree", "shutil.copymode", "shutil.copystat")
_DELETIONS = ("os.remove", "os.rmdir", "shutil.rmtree", "shutil.move", "os.rename")
_LISTINGS = ("os.scandir", "os.listdir")


class SandboxViolation(RuntimeError):
    pass


def _path_of(arg):
    if isinstance(arg, int) or arg is None:
        return None
    try:
        return os.fspath(arg)
    except TypeError:
        return None


def install(workdir: str) -> None:
    workdir = os.path.realpath(workdir)
    allowed_roots = (workdir, os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix))
    readable_roots = allowed_roots + tuple(
        os.path.realpath(p) for p in sys.path if p and os.path.isdir(p)
    )

    def inside(path, roots):
        if path is None:
            return True  # descriptor-based calls: leave to the OS limits
        real = os.path.realpath(os.path.join(workdir, path)) if not os.path.isabs(path) else os.path.realpath(path)
        return any(real == r or real.startswith(r + os.sep) for r in roots)

    def hook(event, args):
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            if mode is None:  # os.open(): judge by flags
                writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            else:
                writing = any(c in mode for c in "wax+")
            if writing and _path_of(path) not in ("/dev/null", "/dev/stdout", "/dev/stderr") \
                    and not inside(_path_of(path), (workdir,)):
                raise SandboxViolation(f"write outside the sandbox: {path!r}")
            return
        if event == "os.chdir":  # pytest returns to its start path; nowhere else
            if not inside(_path_of(args[0] if args else None), (workdir,)):
                raise SandboxViolation(f"chdir outside the sandbox: {args[0]!r}")
            return
        if event in _LISTINGS:
            if not inside(_path_of(args[0] if args else None), readable_roots):
                raise SandboxViolation(f"{event} outside the sandbox: {args[0]!r}")
            return
        if event in _MUTATIONS:
            paths = [_path_of(a) for a in args[:2]]
            if event in _DELETIONS or not all(inside(p, (workdir,)) for p in paths if p is not None):
                if not all(inside(p, (workdir,)) for p in paths if p is not None):
                    raise SandboxViolation(f"{event} outside the sandbox: {paths!r}")
            return
        if event.startswith(_BLOCKED_PREFIXES):
            raise SandboxViolation(f"blocked operation: {event}")

    sys.addaudithook(hook)


def read_nonce(workdir: str) -> str:
    path = os.path.join(workdir, ".nonce")
    with open(path, encoding="utf-8") as stream:
        nonce = stream.read().strip()
    os.remove(path)
    return nonce


def pass_line(nonce: str) -> str:
    return f"__CODE_REWARD_PASS__ {nonce}"
