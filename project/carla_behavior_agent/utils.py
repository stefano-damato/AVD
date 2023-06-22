import socket
import numpy as np
import requests
import threading 
import time
import base64
import matplotlib
from matplotlib import pyplot as plt
import shapely.geometry
import shapely.affinity

def threaded(func):
    def wrapper(*k, **kw):
        thread = threading.Thread(target=func,args=k,kwargs=kw, daemon=True)
        thread.start()
        return thread
    return wrapper

class Plot():
    def __init__(self, filename, plotname) -> None:
        self.filename = filename
        self.plotname = plotname
        
    def plot(self):
        x_data = []
        
        cur_data = []
        target_data = []
        
        with open(self.filename, "r") as fp:
            for line in fp:
                timestamp, speed, target = line.split(";")
                x_data.append(float(timestamp))
                cur_data.append(float(speed))
                target_data.append(float(target))
            fp.close()
        plt.plot(x_data, cur_data)
        plt.plot(x_data, target_data)
        plt.savefig(self.plotname)
        
        
        
class Streamer():

    def __init__(self, IP):
        self.run = True
        self.verbose = False

        self.data ={
                "url" : "http://"+IP+":8888/new_frame",
                "RGB" : {
                    "frame_lock" : threading.Lock(),
                    "frame" : None,
                    "update" : False
                },
                "Depth" : {
                    "frame_lock" : threading.Lock(),
                    "frame" : None,
                    "update" : False
                },
                "BEV" : {
                    "frame_lock" : threading.Lock(),
                    "frame" : None,
                    "update" : False
                },
                "Controls" : {
                    "data_lock" : threading.Lock(),
                    "data" : None,
                    "update" : False
                }
        }

        # launch the streamer thread
        threading.Thread(target=self.sendRGBImage, name = "RGBStreamer").start()
        threading.Thread(target=self.sendDepthImage, name = "DepthStreamer").start()
        threading.Thread(target=self.sendBEVImage, name = "BEVStreamer").start()
        threading.Thread(target=self.sendControlsData, name = "ControlsDataStreamer").start()
        

    def sendRGBImage(self):
        self.__sendImage("RGB")

    def sendDepthImage(self):   
        self.__sendImage("Depth")
    
    def sendBEVImage(self):   
        self.__sendImage("BEV")
    
    def sendControlsData(self):   
        self.__sendData("Controls")
    
    def __sendData(self, datatype):
        while self.run:
            try:
                if self.data[datatype]["data"] is not None and self.data[datatype]["update"]:
                    if self.verbose:
                        print("acquire")
                    self.data[datatype]["data_lock"].acquire()
                    if self.verbose:
                        print("send_image")

                    data = {
                        "type" : datatype, 
                        "data" : self.data[datatype]["data"]
                    }
                    requests.post(self.data["url"], json=data, timeout=10)

                    if self.verbose:
                        print("post_ok")
                    self.data[datatype]["update"] = False
                    self.data[datatype]["data_lock"].release()
                    if self.verbose:
                        print("sent")
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(e)
                if self.data[datatype]["data_lock"].locked():
                    self.data[datatype]["data_lock"].release() 

    def __sendImage(self, datatype):
        ''''
        Take the objects image and send it
        '''
        while self.run:
            try:
                if self.data[datatype]["frame"] is not None and self.data[datatype]["update"]:
                    if self.verbose:
                        print("acquire")
                    self.data[datatype]["frame_lock"].acquire()
                    if self.verbose:
                        print("send_image")

                    data = {
                        "type" : datatype, 
                        "data" : {
                            "encode" : base64.b64encode(self.data[datatype]["frame"].tobytes()).decode('utf-8'),
                            "dtype" : str(self.data[datatype]["frame"].dtype),
                            "shape" : self.data[datatype]["frame"].shape
                        }
                    }
                    requests.post(self.data["url"], json=data, timeout=10)

                    if self.verbose:
                        print("post_ok")
                    self.data[datatype]["update"] = False
                    self.data[datatype]["frame_lock"].release()
                    if self.verbose:
                        print("sent")
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(e)
                if self.data[datatype]["frame_lock"].locked():
                    self.data[datatype]["frame_lock"].release()
        print("---- Stream finished ----")
    
    def send_data(self, datatype, data):
        self.data[datatype]["data_lock"].acquire()
        self.data[datatype]["data"] = data
        self.data[datatype]["update"] = True
        self.data[datatype]["data_lock"].release()

    def send_frame(self, datatype, frame):
        self.data[datatype]["frame_lock"].acquire()
        self.data[datatype]["frame"] = frame
        self.data[datatype]["update"] = True
        self.data[datatype]["frame_lock"].release()

if __name__ == "__main__":
    speed_plot = Plot("./userCode/speed.txt", "./userCode/speedplot.png")
    speed_plot.plot()


class RotatedRectangle(object):

    """
    This class contains method to draw rectangle and find intersection point.
    """

    def __init__(self, c_x, c_y, width, height, angle):
        self.c_x = c_x
        self.c_y = c_y
        self.w = width      # pylint: disable=invalid-name
        self.h = height     # pylint: disable=invalid-name
        self.angle = angle

    def get_contour(self):
        """
        create contour
        """
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.c_x, self.c_y)

    def intersection(self, other):
        """
        Obtain a intersection point between two contour.
        """
        return self.get_contour().intersection(other.get_contour())
