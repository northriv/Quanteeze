# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""System memory, cgroup, and thermal pressure readings and output-neutral thermal throttling (`quanteeze.guards`).

Readings come from the kernel without root:
- On macOS: thermal pressure from notify key `com.apple.system.thermalpressurelevel` (0 nominal .. 4),
  memory pressure from sysctl `kern.memorystatus_vm_pressure_level` (1 normal, 2 warning, 4 critical),
  and swap from `vm.swapusage`.
- On Linux: swap from `/proc/meminfo` (`SwapTotal` less `SwapFree`), memory pressure from `/proc/pressure/memory`
  (PSI) and `MemAvailable`, and cgroup memory limits (`cgroup_memory`: cgroup v2 and v1, excluding file cache).

`Throttle` rests before each forward pass while thermal pressure is at `at` (default 2, heavy) or above, in
proportion to how long the previous pass took, so the GPU works `duty` of the time without changing generated
tokens.
"""
import ctypes
import ctypes.util
import os
import struct
import sys
import time
from pathlib import Path

from . import variables

THERMAL_KEY = b'com.apple.system.thermalpressurelevel'
PSI_WARNING = 10.0              # % of the last 10 s some task stalled on memory: warning
PSI_CRITICAL = 10.0             # % of it every task did: critical
AVAILABLE_WARNING = 0.05        # MemAvailable's share of MemTotal under which it is a warning
HOST_LEAVES = 16e9              # off macOS, what QUANTEEZE_KEEP_GB=auto leaves for page cache and OS (up to 16 GB)
AVAILABLE_CRITICAL = 0.02       # and critical
WORKING_SET_SHARE = 0.75        # share of memory taken as the working set where Metal recommends none
_OTHERS = []                    # bytes of the GPU's memory others held when first seen (note_others)
_DARWIN_LIB = []                # [CDLL | None], opened once per process on macOS
_THERMAL_TOKENS = {}            # id(lib) -> (lib, c_int): one notify_register_check token per lib

_UNSEEN = set()                 # the cgroups found unseen, each said once (_unseen)


def _darwin_lib():
    if not _DARWIN_LIB:
        try:
            _DARWIN_LIB.append(ctypes.CDLL(ctypes.util.find_library('System') or '/usr/lib/libSystem.B.dylib'))
        except OSError:
            _DARWIN_LIB.append(None)
    return _DARWIN_LIB[0]


def _unseen(d):
    """Warn once when the process's memory cgroup `d` is named in `/proc` but not mounted under `/sys/fs/cgroup`."""
    if d in _UNSEEN:
        return
    _UNSEEN.add(d)
    print(f'quanteeze: the memory cgroup {d} cannot be read, so its limit is not seen (in a container, '
          f'bind /sys/fs/cgroup: singularity exec --bind /sys/fs/cgroup)', file=sys.stderr, flush=True)


class Readings:
    """The system's readings: on macOS through libSystem, on Linux from `proc` (`/proc`); None
    elsewhere or on failure."""

    def __init__(self, proc=None, cgroup=None, lib=None):
        self._lib = lib if lib is not None else (_darwin_lib() if sys.platform == 'darwin' else None)
        self._proc = Path(proc) if proc else Path('/proc') if sys.platform.startswith('linux') else None
        self._cgroup = Path(cgroup) if cgroup else Path('/sys/fs/cgroup') if self._proc == Path('/proc') else None
        self._token = None

    def cgroup_memory(self):
        """`(limit, used)` bytes of the tightest memory limit of the process's cgroup and those above
        it, `used` not counting the file cache, or `None` where no cgroup limits the memory."""
        if self._cgroup is None:
            return None
        try:
            lines = (self._proc / 'self' / 'cgroup').read_text().splitlines()
        except (OSError, TypeError):
            return None
        where = None
        for line in lines:
            _, controllers, path = line.split(':', 2)
            if 'memory' in controllers.split(','):
                where = (self._cgroup / 'memory', path, 'memory.limit_in_bytes', 'memory.usage_in_bytes',
                         ('total_active_file', 'total_inactive_file'))
                break
            if controllers == '' and where is None:
                where = (self._cgroup, path, 'memory.max', 'memory.current', ('active_file', 'inactive_file'))
        if where is None:
            return None
        root, path, limit_name, used_name, cache_names = where
        best = None
        d = root / path.strip('/')
        if not d.is_dir():
            _unseen(d)
            return None
        while True:
            try:
                text = (d / limit_name).read_text().strip()
                limit = None if text == 'max' else int(text)
                if limit is not None and limit < 1 << 60 and (best is None or limit < best[0]):
                    used = int((d / used_name).read_text())
                    stat = dict(line.split() for line in (d / 'memory.stat').read_text().splitlines()
                                if len(line.split()) == 2)
                    best = (limit, max(0, used - sum(int(stat.get(name, 0)) for name in cache_names)))
            except (OSError, ValueError):
                pass
            if d == root or root not in d.parents:
                break
            d = d.parent
        return best

    def _meminfo(self):
        """`/proc/meminfo`, `{name: bytes}`, or `None`."""
        try:
            text = (self._proc / 'meminfo').read_text()
        except (OSError, TypeError):
            return None
        out = {}
        for line in text.splitlines():
            name, _, rest = line.partition(':')
            fields = rest.split()
            if fields and fields[0].isdigit():
                out[name] = int(fields[0]) * (1024 if fields[1:] == ['kB'] else 1)
        return out

    def _psi(self):
        """Memory PSI's last 10 s, `{'some' | 'full': % of the time}`, or `None`."""
        try:
            text = (self._proc / 'pressure' / 'memory').read_text()
        except (OSError, TypeError):
            return None
        out = {}
        for line in text.splitlines():
            kind, *fields = line.split()
            values = dict(f.split('=', 1) for f in fields)
            if 'avg10' in values:
                out[kind] = float(values['avg10'])
        return out or None

    def _sysctl(self, name, size):
        if self._lib is None:
            return None
        buf, length = ctypes.create_string_buffer(size), ctypes.c_size_t(size)
        if self._lib.sysctlbyname(name, buf, ctypes.byref(length), None, ctypes.c_size_t(0)) != 0:
            return None
        return buf.raw[:length.value]

    def thermal(self):
        """Thermal pressure: 0 nominal, 1 moderate, 2 heavy, 3 trapping, 4 sleeping."""
        if self._lib is None:
            return None
        if self._token is None:
            entry = _THERMAL_TOKENS.get(id(self._lib))
            if entry is None:
                token = ctypes.c_int()
                if self._lib.notify_register_check(THERMAL_KEY, ctypes.byref(token)) != 0:
                    return None
                entry = _THERMAL_TOKENS[id(self._lib)] = (self._lib, token)
            self._token = entry[1]
        state = ctypes.c_uint64()
        if self._lib.notify_get_state(self._token, ctypes.byref(state)) != 0:
            return None
        return int(state.value)

    def close(self):
        """Release any thermal notify token registered for `self._lib`."""
        entry = _THERMAL_TOKENS.pop(id(self._lib), None) if self._lib is not None else None
        if entry is not None and hasattr(self._lib, 'notify_cancel'):
            self._lib.notify_cancel(entry[1])
        self._token = None

    def pressure(self):
        """Memory pressure: 1 normal, 2 warning, 4 critical."""
        if self._lib is None and self._proc is not None:
            return self._linux_pressure()
        raw = self._sysctl(b'kern.memorystatus_vm_pressure_level', 4)
        return struct.unpack('<i', raw[:4])[0] if raw and len(raw) >= 4 else None

    def _linux_pressure(self):
        info, psi = self._meminfo() or {}, self._psi() or {}
        available = info['MemAvailable'] / info['MemTotal'] if info.get('MemTotal') and 'MemAvailable' in info \
            else None
        limited = self.cgroup_memory()
        if limited is not None:
            left = (limited[0] - limited[1]) / limited[0]
            available = left if available is None else min(available, left)
        if available is None and not psi:
            return None
        if psi.get('full', 0) >= PSI_CRITICAL or (available is not None and available < AVAILABLE_CRITICAL):
            return 4
        if psi.get('some', 0) >= PSI_WARNING or (available is not None and available < AVAILABLE_WARNING):
            return 2
        return 1

    def swap_used(self):
        """Bytes of swap in use."""
        if self._lib is None and self._proc is not None:
            info = self._meminfo() or {}
            return info['SwapTotal'] - info['SwapFree'] if 'SwapTotal' in info and 'SwapFree' in info else None
        raw = self._sysctl(b'vm.swapusage', 32)
        return struct.unpack('<QQQ', raw[:24])[2] if raw and len(raw) >= 24 else None

    def mlx_held(self):
        """Bytes MLX holds, active and cached."""
        try:
            import mlx.core as mx
        except ImportError:
            return None
        return mx.get_active_memory() + mx.get_cache_memory()

    def footprint(self):
        """Bytes this process holds: on macOS its physical footprint (`proc_pid_rusage`), on Linux
        what is resident and swapped (`VmRSS` + `VmSwap`) less the file cache it maps (`_mapped_files`)."""
        if self._lib is not None:
            buf = ctypes.create_string_buffer(96)                    # rusage_info_v0
            if self._lib.proc_pid_rusage(os.getpid(), 0, buf) != 0:
                return None
            return struct.unpack_from('<Q', buf.raw, 16 + 7 * 8)[0]  # ri_phys_footprint
        try:
            text = (self._proc / 'self' / 'status').read_text()
        except (OSError, TypeError):
            return None
        fields = dict(line.split(':', 1) for line in text.splitlines() if ':' in line)
        kb = [int(fields[n].split()[0]) for n in ('VmRSS', 'VmSwap') if n in fields]
        return (sum(kb) - self._mapped_files()) * 1024 if kb else None

    def _mapped_files(self):
        """kB of regular files' pages this process maps, resident and its own only as file cache."""
        try:
            text = (self._proc / 'self' / 'smaps').read_text()
        except (OSError, TypeError):
            return 0
        total, rss, regular = 0, 0, False
        for line in text.splitlines():
            if line.startswith('Rss:'):
                rss = int(line.split()[1])
            elif line.startswith('Anonymous:'):
                if regular:
                    total += rss - int(line.split()[1])
            elif line and line[0] in '0123456789abcdef' and '-' in line.split(' ', 1)[0]:
                parts = line.split(None, 5)
                path = parts[5] if len(parts) > 5 else ''
                regular = path.startswith('/') and not path.startswith('/dev/')
                rss = 0
        return max(0, total)

    def others(self, least=1e9):
        """`[(bytes, pid, name)]` of other processes holding at least `least` bytes, largest first."""
        out = []
        me = os.getpid()
        if self._lib is not None:
            n = self._lib.proc_listallpids(None, 0)
            if n <= 0:
                return []
            pids = (ctypes.c_int * (n + 64))()
            n = self._lib.proc_listallpids(pids, ctypes.sizeof(pids))
            buf, name = ctypes.create_string_buffer(96), ctypes.create_string_buffer(256)
            for pid in pids[:max(0, n)]:
                if pid <= 0 or pid == me or self._lib.proc_pid_rusage(pid, 0, buf) != 0:
                    continue
                held = struct.unpack_from('<Q', buf.raw, 16 + 7 * 8)[0]
                if held >= least:
                    self._lib.proc_name(pid, name, 256)
                    out.append((held, pid, name.value.decode(errors='replace') or '?'))
        elif self._proc is not None:
            for d in self._proc.iterdir():
                if not d.name.isdigit() or int(d.name) == me:
                    continue
                try:
                    fields = dict(line.split(':', 1) for line in (d / 'status').read_text().splitlines() if ':' in line)
                except OSError:
                    continue
                kb = sum(int(fields[k].split()[0]) for k in ('VmRSS', 'VmSwap') if k in fields)
                if kb * 1024 >= least:
                    out.append((kb * 1024, int(d.name), fields.get('Name', '?').strip()))
        return sorted(out, reverse=True)

    def memory_total(self):
        """Bytes of the machine's memory: hw.memsize; on Linux MemTotal, or the memory cgroup's
        limit where that is less."""
        if self._lib is not None:
            raw = self._sysctl(b'hw.memsize', 8)
            return struct.unpack('<Q', raw[:8])[0] if raw and len(raw) >= 8 else None
        total = (self._meminfo() or {}).get('MemTotal')
        limited = self.cgroup_memory()
        if limited is not None:
            total = min(total, limited[0]) if total else limited[0]
        return total

    def swap_disk_free(self):
        """Bytes free on the disk macOS swaps to (/System/Volumes/VM), or None."""
        if self._lib is None:
            return None
        try:
            st = os.statvfs('/System/Volumes/VM' if os.path.isdir('/System/Volumes/VM') else '/')
        except OSError:
            return None
        return st.f_bavail * st.f_frsize


def _mlx_held():
    """Bytes MLX holds now, its arrays and its freed buffers kept (0 where it cannot be asked)."""
    try:
        import mlx.core as mx
        return mx.get_active_memory() + mx.get_cache_memory()
    except (ImportError, RuntimeError, AttributeError):
        return 0


def note_others(info=None, held=None):
    """What others hold of the GPU's memory -- other programs, a desktop, the driver's context -- noted
    once, as early as may be (serve, before a model is loaded): its whole less what is free less what
    MLX holds itself (`held`), never under 0.  Later notes keep the first.  Returns it, or None where the
    GPU has no memory of its own to ask about (a Mac, the CPU)."""
    if _OTHERS:
        return _OTHERS[0]
    if info is None:
        try:
            import mlx.core as mx
            info = mx.device_info()
        except (ImportError, RuntimeError):
            return None
    total, free = info.get('total_memory'), info.get('free_memory')
    if info.get('max_recommended_working_set_size') or not total or free is None:
        return None
    _OTHERS.append(max(0, int(total - free - (_mlx_held() if held is None else held))))
    return _OTHERS[0]


def working_set_share():
    """QUANTEEZE_WORKING_SET_SHARE as a share, None for auto (recommended_budget)."""
    text = (variables.get('QUANTEEZE_WORKING_SET_SHARE') or 'auto').strip().lower()
    if text == 'auto':
        return None
    share = variables.number('QUANTEEZE_WORKING_SET_SHARE')
    if not 0 < share <= 1:
        raise ValueError(f'QUANTEEZE_WORKING_SET_SHARE is a share of the memory, over 0 and at most 1: {share:g}')
    return share


def recommended_budget(info=None, proc=None, cgroup=None):
    """What this machine's MLX work should keep to, in bytes, or None.  On a Mac, what Metal
    recommends (`max_recommended_working_set_size`).  On a GPU with memory of its own (CUDA:
    `device_info`'s `free_memory`), all of it but what others held when first seen (`note_others`).
    Elsewhere `QUANTEEZE_WORKING_SET_SHARE` (`WORKING_SET_SHARE`) of the memory MLX computes in."""
    given = info is not None
    if info is None:
        try:
            import mlx.core as mx
            info = mx.device_info()
        except (ImportError, RuntimeError):
            info = {}
    if info.get('max_recommended_working_set_size'):
        return info['max_recommended_working_set_size']
    share = working_set_share()
    if share is None and info.get('total_memory') and info.get('free_memory') is not None:
        others = (max(0, int(info['total_memory'] - info['free_memory'])) if given
                  else note_others(info))
        return int(info['total_memory'] - others)
    share = WORKING_SET_SHARE if share is None else share
    total = info.get('total_memory')
    if not total:
        readings = Readings(proc, cgroup)
        total = (readings._meminfo() or {}).get('MemTotal')
        limited = readings.cgroup_memory()
        if limited is not None:
            total = min(total, limited[0]) if total else limited[0]
    return int(total * share) if total else None


HOLD_INTAKE = [False]            # critical system memory pressure (Keeper._system): intake takes one row at a time

VMSTAT_KEYS = (
    'compact_stall', 'compact_fail', 'pswpin', 'pswpout', 'pgmajfault',
    'workingset_refault_file', 'pgsteal_direct', 'allocstall_normal', 'allocstall_movable',
)


def host_memory_state():
    """Host memory telemetry dict on Linux (`/proc/meminfo`, `/proc/buddyinfo`, `/proc/vmstat`, PSI), else None."""
    if not sys.platform.startswith('linux'):
        return None
    try:
        info = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            k, v = line.split(':', 1)
            info[k] = int(v.split()[0]) * 1024
        free_pages = large_pages = 0
        for line in Path('/proc/buddyinfo').read_text().splitlines():
            counts = [int(x) for x in line.split()[4:]]
            for order, n in enumerate(counts):
                free_pages += n << order
                if order >= 9:
                    large_pages += n << order
        vm = {}
        for line in Path('/proc/vmstat').read_text().splitlines():
            k, _, v = line.partition(' ')
            if k in VMSTAT_KEYS:
                vm[k] = int(v)

        def psi(kind):
            try:
                first = Path(f'/proc/pressure/{kind}').read_text().splitlines()[0]
                return float(dict(f.split('=') for f in first.split()[1:])['avg10'])
            except (OSError, KeyError, ValueError, IndexError):
                return None

        return dict(
            free_gb=info.get('MemFree', 0) / 1e9,
            available_gb=info.get('MemAvailable', 0) / 1e9,
            swap_free_gb=info.get('SwapFree', 0) / 1e9,
            swap_gb=info.get('SwapTotal', 0) / 1e9,
            shmem_gb=info.get('Shmem', 0) / 1e9,
            large_free_share=large_pages / free_pages if free_pages else 0.0,
            memory_psi=psi('memory'),
            io_psi=psi('io'),
            **vm,
        )
    except (OSError, ValueError):
        return None


STRAIN = dict(memory_psi=10.0, io_psi=20.0, swapout_mb_s=4.0)
STRAIN_POLLS = 2
STRAIN_CALM = 3


class Strain:
    """Tracks Linux host memory strain using PSI, swapout rate, and available-memory floor."""

    def __init__(self, marks=None, polls=STRAIN_POLLS, state=host_memory_state, clock=time.monotonic, floor_gb=None,
                 calm=STRAIN_CALM):
        self.marks, self.polls, self.state, self.clock = dict(marks or STRAIN), polls, state, clock
        self.floor_gb, self.calm = floor_gb, calm
        self.under = 0
        self.last = None
        self.over = 0
        self.strained = False
        self.times = 0
        self.seconds = 0.0
        self.since = None
        self.worst = {}

    def sample(self):
        """Sample host memory state and return `(strained, rates_dict)` or None if unavailable."""
        now, s = self.clock(), self.state()
        if s is None:
            return None
        rates = {
            'memory_psi': s.get('memory_psi') or 0.0,
            'io_psi': s.get('io_psi') or 0.0,
            'available_gb': round(s['available_gb'], 1),
        }
        if self.last is not None and now > self.last[0]:
            dt, was = now - self.last[0], self.last[1]
            rates['swapout_mb_s'] = round((s.get('pswpout', 0) - was.get('pswpout', 0)) * 4096 / 1e6 / dt, 2)
            rates['allocstall_s'] = round(
                (s.get('allocstall_normal', 0) + s.get('allocstall_movable', 0)
                 - was.get('allocstall_normal', 0) - was.get('allocstall_movable', 0)) / dt,
                1,
            )
        self.last = (now, s)
        over = (
            any(rates.get(k, 0.0) >= v for k, v in self.marks.items())
            or (self.floor_gb is not None and s['available_gb'] < self.floor_gb)
        )
        self.over = self.over + 1 if over else 0
        self.under = 0 if over else self.under + 1
        became = not self.strained and self.over >= self.polls
        if became:
            self.strained, self.times, self.since = True, self.times + 1, now
        elif self.strained and self.under >= self.calm:
            self.strained = False
            self.seconds += now - self.since
            self.since = None
        if self.strained:
            for k in self.marks:
                self.worst[k] = max(self.worst.get(k, 0.0), rates.get(k, 0.0))
        rates['became'] = became
        return self.strained, rates

    def summary(self):
        held = self.seconds + (self.clock() - self.since if self.since is not None else 0.0)
        return (
            f'the host strained {self.times} times, {held:.0f} s in all'
            + (f' (at worst {", ".join(f"{k} {v:g}" for k, v in self.worst.items())})' if self.times else '')
        )


def _gb(n):
    return '?' if n is None else f'{n / 1e9:.1f}'


class Keeper:
    """A server's own memory, kept to: it stops itself before the machine runs out."""

    def __init__(self, limit, stop_at=None, swap_growth=8e9, disk_floor=20e9, idle_growth=4e9, step=4e9,
                 poll=2.0, readings=None, idle=lambda: False, shed=None, stop=None, exit=None, stacks=None,
                 log=None, system=True, strain=None, relieve=None):
        self.readings = readings or Readings()
        self.strain = strain
        self.relieve = relieve
        self.limit, self.stop_at = limit, max(stop_at, limit) if stop_at is not None else None
        self.swap_growth, self.disk_floor, self.idle_growth, self.step, self.poll = \
            swap_growth, disk_floor, idle_growth, step, poll
        self.idle, self.shed, self.stop, self.exit, self.stacks = idle, shed, stop, exit, stacks
        self.log = log or (lambda text: print(text, file=sys.stderr, flush=True))
        self.swap_start = self.readings.swap_used()
        self.over = False
        self.idle_from = None
        self.idle_said = False
        self.said = None
        self.stopped = None
        self.sheds = 0
        self.peak = 0
        self.system = system
        self.pressed = 1
        self.pressings = 0
        self._was_strained = False
        HOLD_INTAKE[0] = False
        self._stop_event = None
        self._thread = None

    @classmethod
    def from_env(cls, **kwargs):
        """A keeper set by QUANTEEZE_KEEP_GB, QUANTEEZE_KEEP_FREE_GB, QUANTEEZE_KEEP_SWAP_GB,
        QUANTEEZE_KEEP_DISK_GB, QUANTEEZE_KEEP_PRESSURE, QUANTEEZE_KEEP_STRAIN, and QUANTEEZE_KEEP_CUT_GB."""
        value = (variables.get('QUANTEEZE_KEEP_GB') or 'auto').strip().lower()
        if value == 'off':
            return None
        readings = kwargs.get('readings') or Readings()
        kwargs['readings'] = readings
        total = readings.memory_total()
        if value == 'auto':
            limit = (recommended_budget() if readings._lib is not None
                     else total - max(0.1 * total, min(0.25 * total, HOST_LEAVES)) if total else None)
        else:
            limit = float(value) * 1e9
        if not limit:
            return None
        free = variables.number('QUANTEEZE_KEEP_FREE_GB') * 1e9
        kwargs.setdefault('system', not variables.off('QUANTEEZE_KEEP_PRESSURE'))
        if 'strain' not in kwargs and sys.platform.startswith('linux') and not variables.off('QUANTEEZE_KEEP_STRAIN'):
            kwargs['strain'] = Strain(floor_gb=variables.number('QUANTEEZE_KEEP_CUT_GB'))
        return cls(limit, stop_at=total - free if total else None,
                   swap_growth=variables.number('QUANTEEZE_KEEP_SWAP_GB') * 1e9,
                   disk_floor=variables.number('QUANTEEZE_KEEP_DISK_GB') * 1e9, **kwargs)

    def _holds(self, held):
        mlx = self.readings.mlx_held()
        swap, disk = self.readings.swap_used(), self.readings.swap_disk_free()
        rest = None if mlx is None else held - mlx
        return (f'the server holds {_gb(held)} GB (MLX {_gb(mlx)}, the rest {_gb(rest)}), limit {_gb(self.limit)}; '
                f'swap {_gb(swap)} GB'
                + (f', {_gb(swap - self.swap_start)} since start' if swap is not None and self.swap_start is not None
                   else '') + (f'; {_gb(disk)} GB free on the swap\'s disk' if disk is not None else ''))

    def _stop_reason(self, held):
        if self.stop_at is not None and held >= self.stop_at:
            return f'at {_gb(self.stop_at)} GB, QUANTEEZE_KEEP_FREE_GB short of the machine\'s memory'
        if held <= self.limit:
            return None
        swap = self.readings.swap_used()
        if self.swap_growth is not None and swap is not None and self.swap_start is not None \
                and swap - self.swap_start >= self.swap_growth:
            return f'over the limit with the swap grown {_gb(swap - self.swap_start)} GB'
        if self._disk_short() is not None:
            return f'over the limit with {_gb(self._disk_short())} GB left on the swap\'s disk'
        return None

    def _disk_short(self):
        disk = self.readings.swap_disk_free()
        return disk if self.disk_floor is not None and disk is not None and disk <= self.disk_floor else None

    def check(self):
        held = self.readings.footprint()
        if held is None:
            return
        self.peak = max(self.peak, held)
        if self.stopped is not None:
            if self.swap_growth is not None and held - self.stopped >= self.swap_growth and self.exit is not None:
                self.log(f'quanteeze: memory: still growing after the stop, {self._holds(held)}; exiting now')
                self.exit()
            return
        if self.said is None and held > self.limit / 2 or self.said is not None and abs(held - self.said) >= self.step:
            self.said = held
            self.log(f'quanteeze: memory: {self._holds(held)}')
        idle = self.idle()
        if not idle:
            self.idle_from, self.idle_said = None, False
        elif self.idle_from is None:
            self.idle_from = held
        elif not self.idle_said and held - self.idle_from >= self.idle_growth:
            self.idle_said = True
            self.log(f'quanteeze: memory: WARNING grown {_gb(held - self.idle_from)} GB with no request running; '
                     f'{self._holds(held)}; every thread\'s stack follows')
            if self.stacks is not None:
                self.stacks()
        reason = self._stop_reason(held)
        if reason:
            self.stopped = held
            self.log(f'quanteeze: memory: stopping the server, {reason}: {self._holds(held)}')
            if self.stop is not None:
                self.stop()
            return
        if self.system:
            self._system(held)
        if self.strain is not None:
            self._strained(held)
        if held > self.limit and not self.over:
            self.over, self.sheds = True, self.sheds + 1
            if self.shed is not None:
                self.shed()
            after = self.readings.footprint()
            self.log(f'quanteeze: memory: over the limit, MLX\'s freed buffers given back: {self._holds(held)}'
                     + (f'; {_gb(after)} GB after' if after is not None else ''))
        elif held < 0.95 * self.limit:
            self.over = False

    def _strained(self, held):
        """Sample host strain, warn on transitions, and invoke `relieve(rates)` when provided."""
        got = self.strain.sample()
        if got is None:
            return
        strained, rates = got
        if rates.pop('became', False):
            self.log(
                f'quanteeze: memory: WARNING host memory strain (QUANTEEZE_KEEP_STRAIN): '
                f'{", ".join(f"{k} {v:g}" for k, v in rates.items())}; {self._holds(held)}'
            )
        if self.relieve is not None:
            self.relieve(dict(rates, strained=strained))
        if not strained and self._was_strained:
            self.log(
                f'quanteeze: memory: host memory strain eased: '
                f'{", ".join(f"{k} {v:g}" for k, v in rates.items())}'
            )
        self._was_strained = strained

    def _system(self, held):
        """Act on system memory pressure regardless of process footprint."""
        level = self.readings.pressure()
        if level is None:
            return
        if level >= 2 and level > self.pressed:
            self.pressings += 1
            if self.shed is not None:
                self.shed()
            others = ', '.join(f'{name} [{pid}] {_gb(b)} GB' for b, pid, name in self.readings.others(2e9)[:5])
            self.log(
                f'quanteeze: memory: WARNING machine memory pressure is '
                f'{"critical" if level >= 4 else "warning"}: {self._holds(held)}; '
                f'MLX freed buffers released'
                + ('; intake held to one prompt at a time' if level >= 4 else '')
                + (f'; largest other processes: {others}' if others else '')
            )
        elif level == 1 and self.pressed > 1:
            self.log(f'quanteeze: memory: machine memory pressure returned to normal: {self._holds(held)}')
        HOLD_INTAKE[0] = level >= 4
        self.pressed = level

    def start(self):
        import threading
        self._stop_event = threading.Event()

        def run():
            while not self._stop_event.is_set():
                try:
                    self.check()
                except Exception as e:
                    self.log(f'quanteeze: memory: the keeper\'s reading failed ({type(e).__name__}: {e})')
                self._stop_event.wait(self.poll)
            clear = getattr(__import__('mlx.core', fromlist=['clear_streams']), 'clear_streams', None)
            if clear is not None:
                try:
                    clear()
                except Exception:
                    pass

        self._thread = threading.Thread(target=run, name='quanteeze-keeper', daemon=True)
        self._thread.start()
        return self

    def close(self):
        HOLD_INTAKE[0] = False
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None and self._thread.is_alive() and self._thread is not __import__('threading').current_thread():
            self._thread.join(timeout=2.0)

    def state(self):
        held = self.readings.footprint()
        return {'held_gb': None if held is None else round(held / 1e9, 1), 'peak_gb': round(self.peak / 1e9, 1),
                'limit_gb': round(self.limit / 1e9, 1),
                'stop_gb': None if self.stop_at is None else round(self.stop_at / 1e9, 1),
                'stopped': self.stopped is not None}

    def summary(self):
        return (
            f'at most {_gb(self.peak)} GB held, over the limit {self.sheds} times'
            + (f', machine memory pressure raised {self.pressings} times' if self.pressings else '')
            + (f', {self.strain.summary()}' if self.strain is not None else '')
        )


DUTIES = {2: 0.85, 3: 0.5, 4: 0.25}     # thermal level -> the GPU's share of the time


class Throttle:
    """Output-neutral thermal throttling for serving: while the thermal pressure is at `at` or above,
    rest before each forward pass so the GPU works `duty` of the time -- a share, or `{level: share}`,
    the share of the highest level reached (`DUTIES`: 85% at heavy, half at trapping, a quarter at
    sleeping).  A rest is at most `max_rest` seconds, and a pass that comes more than `gap` seconds
    after the one before earns none."""

    def __init__(self, at=2, duty=None, max_rest=2.0, gap=10.0, every=1.0, readings=None, sleep=time.sleep,
                 clock=time.monotonic, log=None):
        duty = dict(DUTIES) if duty is None else duty
        self.at, self.duty, self.max_rest, self.gap, self.every = at, duty, max_rest, gap, every
        self.readings, self.sleep, self.clock = readings or Readings(), sleep, clock
        self.log = log or (lambda text: print(text, file=sys.stderr, flush=True))
        self.rested = 0.0
        self.rests = 0
        self._last = self._read_at = self._level = None
        self._hot = 1.0

    @classmethod
    def from_env(cls, prefix='QUANTEEZE_THROTTLE_', **kwargs):
        """A throttle set by `<prefix>AT` (thermal level; 2; `'off'` for no throttle),
        `<prefix>DUTY` (the GPU's share of the time while hot: one share, or `'<level>:<share>,...'`;
        `DUTIES` by default) and `<prefix>MAX_REST` (seconds; 2)."""
        at = os.environ.get(prefix + 'AT', '2')
        if at.lower() == 'off':
            return None
        text = os.environ.get(prefix + 'DUTY')
        duty = None if not text else {int(float(k)): float(v) for k, v in (p.split(':') for p in text.split(','))} \
            if ':' in text else float(text)
        return cls(at=float(at), duty=duty, max_rest=float(os.environ.get(prefix + 'MAX_REST', '2')), **kwargs)

    def share(self, level):
        """The GPU's share of the time at a thermal level at or above `at`."""
        if not isinstance(self.duty, dict):
            return self.duty
        reached = [k for k in self.duty if k <= level]
        return self.duty[max(reached)] if reached else 1.0

    def _thermal(self, now):
        if self._read_at is None or now - self._read_at >= self.every:
            self._level, self._read_at = self.readings.thermal(), now
        return self._level

    def before(self):
        """Called before each forward pass: rests while it is hot.  Returns the seconds rested."""
        now = self.clock()
        last, self._last = self._last, now
        level = self._thermal(now)
        duty = self.share(level) if level is not None and level >= self.at else 1.0
        if duty != self._hot:
            self._hot = duty
            self.log(f'throttle: thermal pressure {level}; '
                     + (f'the GPU works {duty:.0%} of the time' if duty < 1 else 'full speed again'))
        if duty >= 1 or last is None or now - last > self.gap:
            return 0.0
        rest = min((now - last) * (1 - duty) / duty, self.max_rest)
        self.sleep(rest)
        self.rested += rest
        self.rests += 1
        self._last = self.clock()
        return rest

    def summary(self):
        return f'rested {self.rests} times, {self.rested:.0f} s in all'


def install_throttle(module, throttle):
    """Make `module` (a language model) call `throttle.before()` at the start of every forward pass."""
    base = type(module)

    def __call__(self, *args, **kwargs):
        throttle.before()
        return base.__call__(self, *args, **kwargs)

    module.__class__ = type(f'Throttled{base.__name__}', (base,), {'__call__': __call__})
    return module
