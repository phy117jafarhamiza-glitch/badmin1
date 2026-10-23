"""محاكٍ بسيط لواجهة Streamlit لاختبار تنفيذ الصفحات بدون تثبيت المكتبة (يُستخدم في الاختبارات فقط).

لا يغني عن تجربة التطبيق الفعلية، لكنه يكشف أخطاء الأسماء والمنطق وتدفق البيانات.
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager


class RerunException(Exception):
    pass


class _Ctx:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self._st

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._st, name)


class _Progress:
    def progress(self, *a, **k):
        pass

    def empty(self):
        pass


class FakeSt:
    def __init__(self):
        self.session_state = {}
        self.overrides = {}
        self.calls = []
        self.secrets = {}
        self.__version__ = "fake"
        self.sidebar = _Ctx(self)
        self.column_config = types.SimpleNamespace(
            CheckboxColumn=lambda *a, **k: ("checkbox", a, k),
            NumberColumn=lambda *a, **k: ("number", a, k))

    # -- helpers
    def _w(self, key, default):
        if key is not None and key in self.overrides:
            return self.overrides[key]
        if key is not None and key in self.session_state:
            return self.session_state[key]
        return default

    def _rec(self, name, *a):
        self.calls.append((name, a[:1]))

    # -- outputs
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def f(*a, **k):
            self._rec(name, *a)
        return f

    def set_page_config(self, **k): pass
    def rerun(self): raise RerunException()
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx(self) for _ in range(n)]
    def tabs(self, names): return [_Ctx(self) for _ in names]
    def expander(self, *a, **k): return _Ctx(self)
    def progress(self, *a, **k): return _Progress()
    def download_button(self, label, data, **k):
        self._rec("download_button", label)
        assert isinstance(data, (bytes, str))
        return False
    def metric(self, label, value, *a, **k):
        assert isinstance(value, (str, int, float)), (label, value)
        self._rec("metric", label, value)
    def cache_data(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda fn: fn

    # -- widgets
    def selectbox(self, label, options, index=0, format_func=str, key=None, **k):
        options = list(options)
        for o in options:
            format_func(o)
        if key in self.overrides:
            return self.overrides[key]
        if key in self.session_state:
            return self.session_state[key]
        return options[index] if options else None
    def radio(self, label, options, index=0, format_func=str, key=None, **k):
        return self.selectbox(label, options, index, format_func, key)
    def text_input(self, label, value="", key=None, **k): return self._w(key, value)
    def text_area(self, label, value="", key=None, **k): return self._w(key, value)
    def number_input(self, label, min_value=None, max_value=None, value=0.0, step=None, key=None, **k):
        return self._w(key, value)
    def slider(self, label, min_value, max_value, value=None, step=None, key=None, **k):
        return self._w(key, value if value is not None else min_value)
    def checkbox(self, label, value=False, key=None, **k): return self._w(key, value)
    def button(self, label, key=None, disabled=False, **k):
        v = bool(self._w(key, False)) and not disabled
        return v
    def file_uploader(self, *a, key=None, **k): return self._w(key, None)
    def data_editor(self, df, **k): return df.copy()
    def stop(self): raise RerunException()


def install() -> FakeSt:
    st = FakeSt()
    mod = types.ModuleType("streamlit")
    for name in dir(FakeSt):
        pass
    mod.__dict__["_fake"] = st

    def getattr_(name):
        return getattr(st, name)
    mod.__getattr__ = getattr_  # type: ignore[attr-defined]
    comp = types.ModuleType("streamlit.components")
    v1 = types.ModuleType("streamlit.components.v1")
    v1.html = lambda html, **k: st.calls.append(("components.html", (html,)))
    comp.v1 = v1
    mod.components = comp
    sys.modules["streamlit"] = mod
    sys.modules["streamlit.components"] = comp
    sys.modules["streamlit.components.v1"] = v1
    return st
