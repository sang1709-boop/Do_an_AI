"""Entrypoint tương thích: python anh_sang.py"""
import runpy
runpy.run_module("app.anh_sang", run_name="__main__", alter_sys=True)
