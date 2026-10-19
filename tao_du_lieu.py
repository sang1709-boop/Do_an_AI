"""Chay tu thu muc goc: python tao_du_lieu.py"""
import runpy

if __name__ == "__main__":
    runpy.run_module("app.tao_du_lieu", run_name="__main__", alter_sys=True)
