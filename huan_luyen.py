"""Entrypoint tương thích: python huan_luyen.py"""
import runpy
runpy.run_module("app.huan_luyen", run_name="__main__", alter_sys=True)
