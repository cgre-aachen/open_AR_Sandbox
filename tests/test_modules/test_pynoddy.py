import pytest
from sandbox.modules.pynoddy import PynoddyModule
import pynoddy
import importlib
#importlib.reload(pynoddy)
import pynoddy.output
import pynoddy.history
import sys, os
import subprocess
from sandbox import _test_data, _package_dir
repository_folder = os.path.abspath(_package_dir + '/../../pynoddy/') + os.sep  # Modify here
example_directory = os.path.abspath(repository_folder + 'examples/') + os.sep
history_file = 'simple_two_faults.his'  # Modify here
history = os.path.abspath(example_directory + history_file)
output_folder = os.path.abspath(_test_data['test'] + 'noddy') + os.sep
output = os.path.abspath(output_folder+'noddy_out')

noddy_exec = 'noddy.exe'


def test_simulation_noddy():
    pynoddy.compute_model(history, output, noddy_path=noddy_exec)


def test():
    p = subprocess.Popen(["echo", "Hello World!"], stdout=subprocess.PIPE)
    stdout, _ = p.communicate()

    assert stdout == b"Hello World!\n"


def test_nody_sandbox():
    noddy = PynoddyModule()
    noddy.update(sb_params)
    pass