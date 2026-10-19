"""Chay tu thu muc goc: python tai_xe.py"""
import runpy

if __name__ == "__main__":
    runpy.run_module("app.tai_xe", run_name="__main__", alter_sys=True)
