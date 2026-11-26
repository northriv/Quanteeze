# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""The server's control routes, guarded by a token, and its settings managed from a browser.

mlx-vlm lets a page of any origin call its server (CORS allow_origins=['*']) and asks for no key
on its management routes unless its API key is set: a page open in the browser could unload the
model or change mlx-vlm's settings.  Middleware here asks every control route for the token in an
X-Quanteeze-Token header, and refuses a request a page of another origin sent (its Origin header
names another host than its Host).  The control routes (GUARDED): mlx-vlm's PATCH /v1/settings,
POST /unload and POST /v1/cache/reset (and their unversioned paths), and /v1/quanteeze/*.
A client's own routes (chat completions, messages, responses) and the read-only ones (/health,
/metrics) are left as they were.  With mlx-vlm's API key set, its routes ask for that too.

The token is QUANTEEZE_CONTROL_TOKEN, else the one ~/.quanteeze/control-token holds (TOKEN_FILE,
readable by the user alone), else a new one written there: made once, it stays from start to start;
deleting the file makes a new one at the next start, and so does a file another user or group could
read.  The page, /quanteeze/settings, holds no secret: the token comes in its URL's fragment
(#token=...), which the browser never sends; the server prints that URL at start.

The settings the page manages are the QUANTEEZE_* variables the server reads (quanteeze.variables,
WHEN 'start', 'load', and 'request').  A change is set in the server's environment at once, takes
effect when the server next reads it (at the next load, after /unload, or the next start), and is
saved to QUANTEEZE_SETTINGS (~/.quanteeze/settings.json; off: not saved).  At start the saved
values are set where the environment sets none: a variable the environment sets wins.

And it switches the model served without a restart (switch_model): the requests that generate
are held off (503, with the load's progress) and the ones running finished, mlx-vlm's model let
go and the new one loaded through quanteeze.serve's hooks, with the settings read at a load
(QUANTEEZE_BITS, QUANTEEZE_FROM_SSD_GB, MTP: quanteeze.variables 'load'), the percent told as it
goes (quanteeze.progress).  From then on every request is served by that model, whatever model
it names.
"""
import asyncio
import json
import os
import secrets
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from . import variables

PAGE_PATH = '/quanteeze/settings'
HOME_PATH = '/quanteeze/'
DEFAULT_PORT = '8765'
HOME_API = '/v1/quanteeze/home/'
HOME_ROUTES = {
    ('GET', 'catalog'),
    ('POST', 'start'),
    ('GET', 'presets'),
    ('POST', 'presets'),
    ('POST', 'download/info'),
    ('POST', 'download'),
    ('GET', 'download'),
    ('POST', 'convert/info'),
    ('POST', 'convert'),
    ('GET', 'convert'),
    ('GET', 'live'),
    ('GET', 'connect'),
    ('POST', 'ple'),
    ('GET', 'ple'),
}
SETTINGS_PATH = '/v1/quanteeze/settings'
MODEL_PATH = '/v1/quanteeze/model'
HEADER = b'x-quanteeze-token'
OURS = {
    ('GET', SETTINGS_PATH),
    ('PATCH', SETTINGS_PATH),
    ('GET', MODEL_PATH),
    ('POST', MODEL_PATH),
} | {(m, HOME_API + r) for m, r in HOME_ROUTES}
GENERATING = {
    '/v1/chat/completions', '/chat/completions',
    '/v1/completions', '/completions',
    '/v1/responses', '/responses',
    '/v1/messages', '/messages',
}
GUARDED = {
    ('PATCH', '/v1/settings'), ('PATCH', '/settings'),
    ('POST', '/unload'), ('POST', '/v1/cache/reset'), ('POST', '/cache/reset'),
} | OURS
TOKEN_FILE = Path.home() / '.quanteeze' / 'control-token'
PAGE = Path(__file__).with_name('control.html')
HOME = Path(__file__).with_name('home.html')
ASSETS = Path(__file__).with_name('assets')
ASSET_ROUTES = {
    '/quanteeze/icon.png': ('quanteeze-icon-512.png', b'image/png'),
    '/quanteeze/logo-light.png': ('quanteeze-logo-transparent.png', b'image/png'),
    '/quanteeze/logo-dark.png': ('quanteeze-logo-dark-transparent.png', b'image/png'),
}
LONGEST = 4096
WORDS = ('auto', 'off')
OWN = ('QUANTEEZE_CONTROL_TOKEN', 'QUANTEEZE_SETTINGS', 'QUANTEEZE_APC_DISK_SECRET')
SAMPLING = {'QUANTEEZE_TEMPERATURE': 'temperature', 'QUANTEEZE_SEED': 'seed'}
CHOICES = {
    'QUANTEEZE_ALLOCATOR': ('auto', 'on', 'off'),
    'QUANTEEZE_BITS': ('1', '2', '3', '4', '8', 'mxfp4', 'q3_k', 'q2_k', 'iq2_xs', 'off'),
    'QUANTEEZE_LOAD_WHOLE': ('on', 'off'),
    'QUANTEEZE_APC_ENCRYPT': ('on', 'off'),
    'QUANTEEZE_APC_READ_BACK_PLAIN': ('on', 'off'),
    'QUANTEEZE_CONTEXT_COMPRESS': ('off', 'recent', 'similar', 'h2o', 'snapkv'),
    'QUANTEEZE_CONTEXT_LAYERS': ('uniform', 'pyramid'),
    'QUANTEEZE_KV_BITS': ('8', 'off'),
    'QUANTEEZE_KV_BITS_SELECTED': ('on', 'off'),
    'QUANTEEZE_KV_BITS_UNPACKED': ('auto', 'on', 'off'),
    'QUANTEEZE_COMPACT': ('on', 'off'),
    'QUANTEEZE_VISION_SSD': ('auto', 'on', 'off'),
    'QUANTEEZE_EMBED_SSD': ('on', 'off'),
    'QUANTEEZE_V41_READS_IMAGES': ('on', 'ssd', 'off'),
    'QUANTEEZE_MTP': ('auto', 'dflash2', 'on', 'off'),
    'QUANTEEZE_MTP_EXACT': ('distribution', 'bits'),
    'QUANTEEZE_NGRAM_SSD': ('auto', 'on', 'off'),
    'QUANTEEZE_NGRAM_ADVISE': ('on', 'off'),
    'QUANTEEZE_HOST_NGRAM': ('on', 'off'),
    'QUANTEEZE_ENGRAM_READ_AHEAD': ('on', 'off'),
    'QUANTEEZE_ENGRAM_IDS_ONCE': ('on', 'off'),
    'QUANTEEZE_MTP_TIMING': ('off', 'wall', 'sync', 'profile'),
    'QUANTEEZE_LOOPS': ('on', 'off'),
    'QUANTEEZE_EXPERT_QMV': ('on', 'off'), 'QUANTEEZE_EXPERT_QMV_VERIFY_FUSED': ('on', 'off'),
    'QUANTEEZE_KQUANT_QMV': ('on', 'off'),
    'QUANTEEZE_HC_KERNELS': ('on', 'off'),
    'QUANTEEZE_V41_HEAD_KERNEL': ('on', 'off'),
    'QUANTEEZE_COMPILE_DECODE': ('on', 'off'),
    'QUANTEEZE_LINEAR_ATTN_KERNELS': ('on', 'off'),
    'QUANTEEZE_SIGMOID_ROUTER_KERNELS': ('on', 'off'),
    'QUANTEEZE_INDEXER_KERNEL': ('on', 'off'),
    'QUANTEEZE_CUDA_QMV': ('on', 'off'),
    'QUANTEEZE_CUDA_QMV_ROWS': ('on', 'off'),
    'QUANTEEZE_CUDA_ATTN_ROWS': ('on', 'off'),
    'QUANTEEZE_CUDA_DELTA': ('on', 'off'),
    'QUANTEEZE_CUDA_EXPERTS': ('on', 'off'),
    'QUANTEEZE_CUDA_ROUTER': ('on', 'off'),
    'QUANTEEZE_NUMPY_HUGE_PAGES': ('on', 'off'),
    'QUANTEEZE_CPU_EXPERTS': ('auto', 'on', 'off'),
    'QUANTEEZE_DEMAND_READ_NATIVE': ('auto', 'on', 'off'),
    'QUANTEEZE_CPU_PRIORITY': ('auto', 'idle', 'nice', 'normal', 'wake'),
    'QUANTEEZE_HOST_EXPERTS_BY': ('auto', 'rows', 'layers'),
    'QUANTEEZE_PARALLEL': ('auto', 'layers', 'off'),
    'QUANTEEZE_PARALLEL_CROSSING': ('refuse', 'raise'),
    'QUANTEEZE_KEEP_PRESSURE': ('on', 'off'),
    'QUANTEEZE_KEEP_STRAIN': ('on', 'off'),
    'QUANTEEZE_REQUEST_LEASES': ('on', 'off'),
    'QUANTEEZE_NATIVE': ('on', 'off'), 'QUANTEEZE_NVRTC_PCH': ('on', 'off'),
    'QUANTEEZE_NGRAM_TOUCH_NATIVE': ('on', 'off'),
    'QUANTEEZE_ENGRAM_HASHES_ON_HOST': ('on', 'off'),
    'QUANTEEZE_V41_ATTENTION_NATIVE': ('auto', 'on', 'off', 'compare'),
    'QUANTEEZE_DSPARK_NATIVE': ('auto', 'on', 'off'),
    'QUANTEEZE_DEMAND_STM': ('auto', 'on', 'shadow', 'off'),
    'QUANTEEZE_REQUEST_THREADS_ANY': ('on', 'off'),
    'QUANTEEZE_REQUEST_THREADS_DRAFT_TOGETHER': ('on', 'off'),
}

_SOURCE = {}
_ENVIRON = {}
_STARTED = {}
_LOADED = [None]
_CHANGED = set()
_SWITCH = {
    'busy': False, 'inflight': 0, 'served': None, 'target': None,
    'state': 'idle', 'error': None, 'loader': None, 'check': None,
    'unloader': None, 'held': None,
}


def served():
    """The settings the server reads and the page manages, in the table's order: not the token (a
    secret) nor where the settings are saved, both set where the server starts."""
    return [s for s in variables.SETTINGS.values() if s.name not in OWN]


def _now(name):
    return variables.value(name) or None


def _snapshot():
    return {s.name: _now(s.name) for s in served()}


# ---------------------------------------------------------------- the saved settings

def saved_path():
    return None if variables.off('QUANTEEZE_SETTINGS') else variables.path('QUANTEEZE_SETTINGS')


def _saved_as_written():
    path = saved_path()
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'{path}: not a JSON object of settings')
    return {k: str(v) for k, v in data.items()}


def saved():
    """{name: value} saved from the page, {} where there are none."""
    return variables.with_new_names(_saved_as_written())


def _save(values):
    path = saved_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(dict(sorted(values.items())), indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def start():
    """At the server's start, before anything reads a setting: the saved values set where the
    environment sets none.  Returns the lines to report."""
    lines = []
    _ENVIRON.clear()
    _ENVIRON.update(os.environ)
    for s in served():
        if _now(s.name):
            _SOURCE[s.name] = 'environment'
    try:
        written = _saved_as_written()
    except ValueError as e:
        return [f'the saved settings are left out: {e}']
    values = variables.with_new_names(written)
    for old in (o for o in variables.RENAMED if written.get(o)):
        read = [f'{n}={values[n]}' for n in variables.RENAMED[old] if values.get(n)]
        lines.append(f'{old} in {saved_path()} is an old name, read as {" and ".join(read) or "nothing"}')
    for name, value in values.items():
        setting = variables.SETTINGS.get(name)
        if setting is None or name in OWN:
            lines.append(f'{name} in {saved_path()} is not a setting the page manages: left out')
        elif _now(name):
            lines.append(f'{name}={value} saved, but the environment sets {name}={_now(name)}: that is served')
        elif value:
            os.environ[name] = value
            _SOURCE[name] = 'saved'
    _STARTED.update(_snapshot())
    return lines


def environ():
    """The environment the server was started with, before the saved values were set in it."""
    return dict(_ENVIRON or os.environ)


def loaded():
    """As the server loads a model: what it reads now ('load' settings take effect) -- kept for
    variables.at_load() so mid-load or mid-request env changes do not drift."""
    _LOADED[0] = _snapshot()
    variables.take_load_snapshot()


def unloaded():
    """Clear the active model's settings snapshot on page unload."""
    _LOADED[0] = None
    variables._AT_LOAD[0] = None


# ---------------------------------------------------------------- the settings, read and changed

def state():
    """What the page shows: every setting the server reads, its value now, where it came from, the
    value in effect (as read at start or at the last load) and whether a change waits for that."""
    try:
        keep = saved()
    except ValueError:
        keep = {}
    out = []
    for s in served():
        now = _now(s.name)
        read = _STARTED if s.when == 'start' or _LOADED[0] is None else _LOADED[0]
        effect = (
            now if s.when == 'request' or (s.when == 'load' and _LOADED[0] is None)
            else read.get(s.name)
        )
        out.append({
            'name': s.name, 'group': s.group, 'when': s.when, 'doc': s.doc, 'default': s.default,
            'choices': list(CHOICES.get(s.name, ())),
            'value': now, 'in_effect': effect,
            'pending': now != effect and (s.when == 'start' or _LOADED[0] is not None),
            'source': 'here' if s.name in _CHANGED else _SOURCE.get(s.name) if now else None,
            'saved': keep.get(s.name), 'environment': _SOURCE.get(s.name) == 'environment',
        })
    return {
        'settings': out,
        'loaded': _LOADED[0] is not None,
        'saved_to': str(saved_path()) if saved_path() else None,
        'groups': list(variables.GROUPS),
    }


def _numeric(text):
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def check(name, value):
    """Why `value` cannot be `name`'s, or None."""
    setting = variables.SETTINGS.get(name)
    if setting is None:
        return 'not a setting (python -m quanteeze.variables lists them)'
    if name in OWN:
        return 'set in the environment the server starts in, not from the page'
    if value is None:
        return None
    if not isinstance(value, str):
        return 'a value is text, as an environment variable\'s'
    if len(value) > LONGEST or any(c in value for c in '\0\r\n'):
        return f'one line of at most {LONGEST} characters'
    if value and _numeric(setting.default) and not _numeric(value) and value.lower() not in WORDS:
        return f'a number, or {" or ".join(WORDS)} (the default is {setting.default})'
    if value and name == 'QUANTEEZE_MTP_BITS' and value.lower() not in WORDS and not (
        value.isdigit() and int(value) > 0
    ):
        return f'a positive whole number, or {" or ".join(WORDS)} (the default is {setting.default})'
    if value and name in CHOICES and value not in CHOICES[name]:
        return f'one of {", ".join(CHOICES[name])}'
    if value and name in ('QUANTEEZE_READ_THREADS', 'QUANTEEZE_CPU_THREADS') and value.lower() != 'auto' and not (
        value.isdigit() and int(value) > 0
    ):
        return 'a positive whole number or auto'
    if value and name == 'QUANTEEZE_REQUEST_THREADS' and not (value.isdigit() and int(value) > 0):
        return 'a positive whole number'
    if value and name == 'QUANTEEZE_PROMPT_MOE_TOKENS' and value.lower() not in WORDS and not (
        value.isdigit() and int(value) > 0
    ):
        return 'a positive whole number, auto, or off'
    if value and name == 'QUANTEEZE_WARM_UP_BATCH' and value.lower() != 'off' and not value.isdigit():
        return 'a whole number (0 or higher), or off'
    if value and name in SAMPLING and not (_numeric(value) and (name != 'QUANTEEZE_SEED' or value.isdigit())):
        return 'a whole number' if name == 'QUANTEEZE_SEED' else 'a number'
    return None


def change(values):
    """Set {name: value} (None or '' unsets it) in the environment now and in the saved settings.
    Returns {'applied': [names], 'rejected': {name: why}, ...state()}."""
    if not isinstance(values, dict):
        raise ValueError('values: a JSON object of {name: value}')
    rejected = {n: why for n, v in values.items() if (why := check(n, v))}
    keep = saved()
    applied = []
    for name, value in values.items():
        if name in rejected:
            continue
        if value:
            os.environ[name] = value
            keep[name] = value
        else:
            os.environ.pop(name, None)
            for old in (o for o, new in variables.RENAMED.items() if name in new):
                os.environ.pop(old, None)
            keep.pop(name, None)
        _CHANGED.add(name)
        applied.append(name)
    if applied:
        _save(keep)
    return {'applied': applied, 'rejected': rejected, **state()}


# ---------------------------------------------------------------- the model, switched while serving

def chosen():
    """The model chosen from the page (every generating request is served by it), or None."""
    return _SWITCH['served']


def current():
    """The model mlx-vlm serves now (its text model's path), or None."""
    try:
        from mlx_vlm.server.runtime import runtime
    except ImportError:
        return None
    cache = runtime.model_cache
    caches = [cache] if isinstance(cache, dict) else list(cache.values()) if cache is not None else []
    for c in caches:
        if c.get('model') is not None and c.get('model_kind', 'text_generation') == 'text_generation':
            return c.get('model_path')
    return None


def serving():
    """The model every request is served by: the one chosen from the page, else the one mlx-vlm
    has loaded, else the one the server started with (None where there is none)."""
    model = _SWITCH['served'] or current()
    if model:
        return model
    started = argument(sys.argv[1:], '--model')
    return str(Path(started).expanduser().resolve()) if started and Path(started).expanduser().is_dir() else started


def model_state():
    """The model served, the switch's state (idle, waiting for the requests running, loading,
    failed), its error, the requests running, the seconds since the last one ended, the load's
    progress, and the uniform bits asked for."""
    from . import progress
    last = _SWITCH.get('last')
    return {
        'model': current(), 'served': chosen(), 'target': _SWITCH['target'], 'state': _SWITCH['state'],
        'held': _SWITCH.get('held'),
        'error': _SWITCH['error'], 'inflight': _SWITCH['inflight'],
        'quiet_s': None if last is None else round(time.time() - last, 1), 'progress': progress.state(),
        'bits': _uniform_bits(), 'from_ssd_gb': variables.get('QUANTEEZE_FROM_SSD_GB'),
    }


def _uniform_bits():
    """The bits or uniform mode QUANTEEZE_BITS asks every routed expert at (quanteeze.serve), else None."""
    value = (variables.get('QUANTEEZE_BITS') or '').strip().lower()
    if value in ('mxfp4', 'q3_k', 'q2_k', 'iq2_xs'):
        return value
    return int(value) if value.isdigit() and int(value) else None


def _checked(target):
    """The settings a load of `target` reads, refused before the model served is let go."""
    if _SWITCH['check'] is not None:
        try:
            _SWITCH['check'](target)
        except (SystemExit, Exception) as e:
            raise ValueError(str(e)) from None


def switch_model(path, wait=0.2, undo=None):
    """Serve the model at `path` from now on, in a thread: the requests that generate are held off
    (503, with the progress), the ones running finished, the model served let go and this one
    loaded (serve's loader: quanteeze's hooks, the settings read at a load); then every request
    is served by it, whatever model it names.  `undo`: `{name: value}` the caller changed for this
    load, restored (and saved) if the async load fails while those keys are still unchanged.
    Returns model_state()."""
    import threading
    target = Path(str(path)).expanduser()
    if not (target / 'config.json').is_file():
        raise ValueError(f'{target}: no config.json there, so no checkpoint to load')
    if _SWITCH['loader'] is None:
        raise ValueError('this server loads no model from the page (quanteeze.serve installs the loader)')
    if _SWITCH['busy']:
        raise ValueError(
            f'unloading {_SWITCH["held"]} already' if _SWITCH['state'] == 'unloading'
            else f'loading {_SWITCH["target"]} already'
        )
    from . import convert
    if convert.running():
        raise ValueError(
            f'converting {Path(convert.state().get("source") or "").name} to MLX: load a model when it ends'
        )
    target = str(target.resolve())
    _checked(target)
    _SWITCH.update(busy=True, state='waiting', error=None, target=target)
    ours = {name: os.environ.get(name, '') for name in (undo or {})}

    def run():
        from . import progress
        try:
            while _SWITCH['inflight'] > 0:
                time.sleep(wait)
            _SWITCH['state'] = 'loading'
            _SWITCH['loader'](target)
            _SWITCH.update(served=target, state='idle', held=None)
        except BaseException as e:
            _SWITCH.update(served=None, state='failed', error=f'{type(e).__name__}: {e}')
            back = {n: v for n, v in (undo or {}).items() if os.environ.get(n, '') == ours[n]}
            if back:
                change(back)
                print(f'quanteeze: the load of {target} failed: {", ".join(sorted(back))} set back to what it was',
                      file=sys.stderr, flush=True)
            if progress.state()['loading']:
                progress.end(_SWITCH['error'])
        finally:
            _SWITCH['busy'] = False

    threading.Thread(target=run, name='quanteeze-model', daemon=True).start()
    return model_state()


def unload_model(wait=0.2):
    """Unload the served model and refuse generation requests (503) until reloaded from the Home page."""
    import threading
    if _SWITCH['unloader'] is None:
        raise ValueError('this server lets no model go from the page (quanteeze.serve installs the unloader)')
    if _SWITCH['busy']:
        raise ValueError(
            f'unloading {_SWITCH["held"]} already' if _SWITCH['state'] == 'unloading'
            else f'loading {_SWITCH["target"]} already'
        )
    target = chosen() or current()
    if target is None:
        raise ValueError('no model is loaded to let go')
    _SWITCH.update(busy=True, state='unloading', error=None, target=None, held=target)

    def run():
        try:
            while _SWITCH['inflight'] > 0:
                time.sleep(wait)
            _SWITCH['unloader']()
            _SWITCH.update(served=None, state='unloaded')
        except BaseException as e:                # noqa: BLE001 -- shown on the page; requests stay refused
            _SWITCH.update(state='failed', error=f'{type(e).__name__}: {e}')
        finally:
            _SWITCH['busy'] = False

    threading.Thread(target=run, name='quanteeze-model', daemon=True).start()
    return model_state()


# ---------------------------------------------------------------- the token

def token():
    """QUANTEEZE_CONTROL_TOKEN, else the one TOKEN_FILE holds (kept()), else a new one, written
    there for the user alone."""
    given = variables.get('QUANTEEZE_CONTROL_TOKEN')
    if given:
        return given
    kept_one = kept()
    if kept_one:
        return kept_one
    made = secrets.token_urlsafe(24)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(made + '\n')
    return made


def kept():
    """The token TOKEN_FILE holds, if the file is this user's, no one else can read it, and the
    token is one token() could have made (32 or more of base64url's characters); else None."""
    try:
        st = os.stat(TOKEN_FILE)
        if st.st_uid != os.getuid() or st.st_mode & 0o077:
            return None
        text = TOKEN_FILE.read_text().strip()
    except OSError:
        return None
    return text if len(text) >= 32 and all(c.isalnum() or c in '-_' for c in text) else None


def page_url(host, port, secret):
    """The page's URL, the token in its fragment; a server on every address is reached at 127.0.0.1."""
    host = '127.0.0.1' if host in (None, '', '0.0.0.0', '::') else host
    host = f'[{host}]' if ':' in host else host
    return f'http://{host}:{port}{PAGE_PATH}#token={secret}'


def argument(argv, name, default=None):
    """A command-line option's value (mlx-vlm's --host, --port), or `default`."""
    return argv[argv.index(name) + 1] if name in argv[:-1] else default


# ---------------------------------------------------------------- the middleware

async def _buffer(receive):
    """Read an ASGI request's full body and return `(bytes, replay_receive)`."""
    chunks = []
    more = True
    while more:
        msg = await receive()
        if msg.get('type') == 'http.request':
            chunks.append(msg.get('body', b''))
            more = bool(msg.get('more_body'))
        else:
            break
    raw = b''.join(chunks)
    sent = [False]

    async def again():
        if not sent[0]:
            sent[0] = True
            return {'type': 'http.request', 'body': raw, 'more_body': False}
        return await receive()

    return raw, again


async def _reply(send, status, body, kind=b'application/json'):
    data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
    await send({'type': 'http.response.start', 'status': status,
                'headers': [(b'content-type', kind), (b'content-length', str(len(data)).encode()),
                            (b'cache-control', b'no-store')]})
    await send({'type': 'http.response.body', 'body': data, 'more_body': False})


def same_origin(headers):
    """Whether a request came from no page, or from a page of the server's own origin."""
    origin = headers.get(b'origin')
    if origin is None:
        return True
    host = headers.get(b'host', b'').decode('latin-1')
    return bool(host) and urlparse(origin.decode('latin-1')).netloc == host


class Middleware:
    """ASGI: GET /quanteeze/settings gives the settings page; GET /quanteeze/ gives the home page;
    every control route (GUARDED) asks for the token and a request of the server's own origin;
    GET /v1/quanteeze/settings gives state(), PATCH it ({"values": {name: value}}) change();
    GET/POST /v1/quanteeze/model likewise.  A request that generates waits for no token: it is
    held off while a model loads, named the model chosen here, and given QUANTEEZE_TEMPERATURE and
    QUANTEEZE_SEED where it names none."""

    def __init__(self, app, secret, page=PAGE):
        self.app, self.secret, self.page = app, secret.encode(), page

    async def __call__(self, scope, receive, send):
        if scope.get('type') != 'http':
            return await self.app(scope, receive, send)
        method, path = scope.get('method'), scope.get('path')
        if method == 'GET' and path == PAGE_PATH:
            return await _reply(send, 200, self.page.read_bytes(), b'text/html; charset=utf-8')
        if method == 'GET' and path in (HOME_PATH, HOME_PATH.rstrip('/')):
            return await _reply(send, 200, HOME.read_bytes(), b'text/html; charset=utf-8')
        if method == 'GET' and path in ASSET_ROUTES:
            fname, kind = ASSET_ROUTES[path]
            return await _reply(send, 200, (ASSETS / fname).read_bytes(), kind)
        if method == 'POST' and path in GENERATING:
            return await self._generate(scope, receive, send)
        if (method, path) not in GUARDED:
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers') or [])
        if not same_origin(headers):
            return await _reply(send, 403, {'error': 'a control route answers pages of the server\'s own origin'})
        if not secrets.compare_digest(headers.get(HEADER, b''), self.secret):
            return await _reply(send, 401, {'error': 'the control token is asked for in X-Quanteeze-Token '
                                                     '(the URL the server printed at start)'})
        if (method, path) not in OURS:
            return await self.app(scope, receive, send)
        raw, _ = await _buffer(receive)
        try:
            body = json.loads(raw or b'{}') if method != 'GET' else {}
            if not isinstance(body, dict):
                raise ValueError('the body: a JSON object')
            if path == SETTINGS_PATH:
                out = state() if method == 'GET' else change(body.get('values'))
            elif path == MODEL_PATH:
                if method == 'GET':
                    out = model_state()
                elif body.get('unload'):
                    out = unload_model()
                else:
                    if 'bits' in body:
                        change({'QUANTEEZE_BITS': str(body['bits']) if body['bits'] else ''})
                    out = switch_model(body.get('path') or chosen() or current() or '')
            elif path.startswith(HOME_API):
                from . import home
                out = await asyncio.to_thread(home.handle, method, path[len(HOME_API):], body)
            else:
                raise ValueError(f'{path}: unknown control route')
        except (ValueError, OSError) as e:
            return await _reply(send, 400, {'error': f'{type(e).__name__}: {e}'})
        return await _reply(send, 200, out)

    async def _generate(self, scope, receive, send):
        """A request that generates: refused (503, with the progress) while a model loads or after
        it is unloaded from the Home page; else named the model served (serving()), with defaults."""
        if _SWITCH.get('held') and (not _SWITCH['busy'] or _SWITCH['state'] == 'unloading'):
            data = json.dumps({
                'error': {
                    'message': f'the model ({_SWITCH["held"]}) was let go from the page: '
                               'load it again there to serve',
                    'type': 'unloaded',
                },
            }).encode()
            await send({'type': 'http.response.start', 'status': 503,
                        'headers': [(b'content-type', b'application/json'),
                                    (b'content-length', str(len(data)).encode())]})
            return await send({'type': 'http.response.body', 'body': data, 'more_body': False})
        if _SWITCH['busy']:
            from . import progress
            p = progress.state()
            data = json.dumps({
                'error': {
                    'message': f'the server is loading {_SWITCH.get("target")}: '
                               f'{p["percent"]:g}% ({p["phase"] or "waiting"})',
                    'type': 'loading',
                },
                'progress': p,
            }).encode()
            await send({'type': 'http.response.start', 'status': 503,
                        'headers': [(b'content-type', b'application/json'),
                                    (b'content-length', str(len(data)).encode()),
                                    (b'retry-after', b'5')]})
            return await send({'type': 'http.response.body', 'body': data, 'more_body': False})
        given = {key: float(v) if key == 'temperature' else int(v)
                 for name, key in SAMPLING.items() if (v := os.environ.get(name))}
        model = serving()
        if model or given:
            scope, receive = await _named(scope, receive, model, given)
        _SWITCH['inflight'] += 1
        try:
            return await self.app(scope, receive, send)
        finally:
            _SWITCH['inflight'] -= 1
            _SWITCH['last'] = time.time()


async def _named(scope, receive, model, defaults=None):
    """A request's scope and receive with its JSON body's 'model' set to `model` (None: as it
    is), and each of `defaults` ({key: value}) set where the body names none."""
    raw, again = await _buffer(receive)
    try:
        body = json.loads(raw)
    except ValueError:
        return scope, again
    if not isinstance(body, dict):
        return scope, again
    new = dict(body, **({'model': model} if model else {}))
    for key, value in (defaults or {}).items():
        if new.get(key) is None:
            new[key] = value
    if new == body:
        return scope, again
    data = json.dumps(new, ensure_ascii=False).encode()
    await again()
    sent = [False]

    async def renamed():
        if not sent[0]:
            sent[0] = True
            return {'type': 'http.request', 'body': data, 'more_body': False}
        return await again()

    headers = [(k, v) for k, v in scope.get('headers') or [] if k != b'content-length']
    return dict(scope, headers=headers + [(b'content-length', str(len(data)).encode())]), renamed


def install(app, argv=None, loader=None, check=None, unloader=None):
    """Guard `app`'s control routes and serve the page; returns the page's URL.  loader(path) loads
    a model in mlx-vlm's place (quanteeze.serve), check(path) refuses settings a load of the
    checkpoint at `path` would refuse, before the model served is let go, and unloader() releases it."""
    argv = sys.argv[1:] if argv is None else argv
    secret = token()
    _SWITCH.update(loader=loader, check=check, unloader=unloader, secret=secret)
    app.add_middleware(Middleware, secret=secret)
    return page_url(argument(argv, '--host'), argument(argv, '--port', DEFAULT_PORT), secret)
