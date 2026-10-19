"""Entrypoint tương thích: python realtime.py"""
import runpy
runpy.run_module("app.realtime", run_name="__main__", alter_sys=True)
