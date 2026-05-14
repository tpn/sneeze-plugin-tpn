import importlib


def test_plugin_package_imports():
    assert importlib.import_module("sneeze.tpn")
