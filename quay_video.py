"""Entrypoint tương thích: python quay_video.py"""
import runpy
runpy.run_module("app.quay_video", run_name="__main__", alter_sys=True)
