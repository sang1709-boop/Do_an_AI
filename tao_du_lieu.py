"""Entrypoint tương thích: python tao_du_lieu.py"""
import runpy
runpy.run_module("app.tao_du_lieu", run_name="__main__", alter_sys=True)
