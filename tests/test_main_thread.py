import sandbox as sb
from sandbox.main_thread import MainThread

projector = sb.Projector(use_panel=False)
sensor = sb.Sensor(name='dummy')

def test_init():
    smain = MainThread(sensor, projector)
    print(smain)

def test_update():
    projector_2 = sb.Projector(use_panel=True)
    sensor_2 = sb.Sensor(name='dummy')
    smain = MainThread(sensor_2, projector_2)
    smain.update()

def test_run_method(): #TODO: stuck in infinite loop
    smain = MainThread(sensor, projector)
    smain.run()
    print('run() working')
    smain.pause()
    print('pause() working')
    smain.resume()
    print('resume() working')
    smain.stop()
    print('stop() working')
