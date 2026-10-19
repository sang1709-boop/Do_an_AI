"""Entrypoint tương thích: python kiem_tra_du_lieu.py"""
import runpy
runpy.run_module("app.kiem_tra_du_lieu", run_name="__main__", alter_sys=True)
