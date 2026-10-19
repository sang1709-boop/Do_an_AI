"""Entrypoint tương thích: python camera_cfg.py"""
import runpy
runpy.run_module("app.camera_cfg", run_name="__main__", alter_sys=True)
