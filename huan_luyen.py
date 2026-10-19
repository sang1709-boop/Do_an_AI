"""Chay tu thu muc goc: python huan_luyen.py"""
import runpy

if __name__ == "__main__":
    runpy.run_module("app.huan_luyen", run_name="__main__", alter_sys=True)
