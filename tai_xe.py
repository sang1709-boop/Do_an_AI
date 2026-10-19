"""Entrypoint tương thích: python tai_xe.py"""
import runpy
runpy.run_module("app.tai_xe", run_name="__main__", alter_sys=True)
