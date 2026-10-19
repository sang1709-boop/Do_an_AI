"""Chay tu thu muc goc: python anh_sang.py"""
import runpy

if __name__ == "__main__":
    runpy.run_module("app.anh_sang", run_name="__main__", alter_sys=True)
