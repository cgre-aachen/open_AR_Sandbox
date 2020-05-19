from bokeh.server.server import Server
import panel as pn
from tornado.ioloop import IOLoop
import os,sys
sys.path.append('../../')

import sandbox as sb
CALIBRATION_FILE = '../calibration_files/my_calibration.json'

#sb.start_server(CALIBRATION_FILE)


sb.calibrate_sandbox()
#pn.serve()

#server = Server(port=5000)
#server.io_loop.start()

if __name__ == '__main__':
    pass
