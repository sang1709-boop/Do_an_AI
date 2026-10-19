"""Chay tu thu muc goc: python realtime.py"""
import runpy

if __name__ == "__main__":
    runpy.run_module("app.realtime", run_name="__main__", alter_sys=True)
