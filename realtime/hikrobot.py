"""Hikrobot MVS adapter; install vendor SDK separately. No vendor code bundled."""
import ctypes
import importlib
import json
import os
from pathlib import Path
import sys
import threading

import numpy as np

_initialization_lock = threading.Lock()
_initialized = False


class MVSCamera:
    def __init__(self, config):
        sdk_path = config.get('sdk_python_path') or os.environ.get('MVS_PYTHON_PATH')
        if not sdk_path:
            raise RuntimeError("Set MVS_PYTHON_PATH to the SDK MvImport directory")
        sys.path.insert(0, str(Path(sdk_path).expanduser()))
        self.sdk = importlib.import_module('MvCameraControl_class')
        self.cam = None
        self.handle_created = self.opened = self.grabbing = False
        sdk = self.sdk
        global _initialized
        with _initialization_lock:
            if not _initialized and hasattr(sdk.MvCamera, 'MV_CC_Initialize'):
                self.check(sdk.MvCamera.MV_CC_Initialize(), 'Initialize')
                _initialized = True
        try:
            devices = sdk.MV_CC_DEVICE_INFO_LIST()
            self.check(sdk.MvCamera.MV_CC_EnumDevices(sdk.MV_GIGE_DEVICE | sdk.MV_USB_DEVICE, devices), 'EnumDevices')
            selected = None
            serial = config.get('serial')
            if not isinstance(serial, str) or not serial:
                raise ValueError("Camera config requires an explicit serial number")
            for index in range(devices.nDeviceNum):
                info = ctypes.cast(devices.pDeviceInfo[index], ctypes.POINTER(sdk.MV_CC_DEVICE_INFO)).contents
                if info.nTLayerType == sdk.MV_GIGE_DEVICE:
                    raw = info.SpecialInfo.stGigEInfo.chSerialNumber
                elif info.nTLayerType == sdk.MV_USB_DEVICE:
                    raw = info.SpecialInfo.stUsb3VInfo.chSerialNumber
                else:
                    continue
                if bytes(raw).split(b'\0')[0].decode(errors='replace') == serial:
                    selected = info
                    break
            if selected is None:
                raise RuntimeError("Requested MVS camera serial is not available")
            self.cam = sdk.MvCamera()
            self.check(self.cam.MV_CC_CreateHandle(selected), 'CreateHandle')
            self.handle_created = True
            self.check(self.cam.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0), 'OpenDevice')
            self.opened = True
            if selected.nTLayerType == sdk.MV_GIGE_DEVICE:
                packet_size = self.cam.MV_CC_GetOptimalPacketSize()
                if packet_size > 0:
                    self.check(self.cam.MV_CC_SetIntValue('GevSCPSPacketSize', packet_size), 'PacketSize')
            mode = config.get('trigger_mode', 'preserve')
            if mode == 'continuous':
                self.check(self.cam.MV_CC_SetEnumValue('TriggerMode', sdk.MV_TRIGGER_MODE_OFF), 'TriggerMode')
            elif mode != 'preserve':
                raise ValueError("trigger_mode must be preserve or continuous")
            # Keep exposure, gain, white balance and pixel format as configured in MVS.
            self.check(self.cam.MV_CC_StartGrabbing(), 'StartGrabbing')
            self.grabbing = True
        except Exception:
            self.release()
            raise

    @staticmethod
    def check(code, operation):
        if code:
            raise RuntimeError(f"MVS {operation} failed: 0x{code & 0xffffffff:08x}")

    def isOpened(self):
        return self.grabbing

    def read(self):
        sdk = self.sdk
        packet = sdk.MV_FRAME_OUT()
        code = self.cam.MV_CC_GetImageBuffer(packet, 1000)
        if code & 0xffffffff == 0x80000007:  # MV_E_NODATA, e.g. waiting for hardware trigger.
            raise TimeoutError("MVS camera has no frame yet")
        self.check(code, 'GetImageBuffer')
        try:
            info = packet.stFrameInfo
            size = info.nWidth * info.nHeight * 3
            output = (ctypes.c_ubyte * size)()
            params = sdk.MV_CC_PIXEL_CONVERT_PARAM()
            params.nWidth, params.nHeight = info.nWidth, info.nHeight
            params.pSrcData, params.nSrcDataLen = packet.pBufAddr, info.nFrameLen
            params.enSrcPixelType = info.enPixelType
            params.enDstPixelType = sdk.PixelType_Gvsp_BGR8_Packed
            params.pDstBuffer = ctypes.cast(output, ctypes.POINTER(ctypes.c_ubyte))
            params.nDstBufferSize = size
            self.check(self.cam.MV_CC_ConvertPixelType(params), 'ConvertPixelType')
            return True, np.ctypeslib.as_array(output).reshape(info.nHeight, info.nWidth, 3).copy()
        finally:
            self.cam.MV_CC_FreeImageBuffer(packet)

    def release(self):
        if self.cam is not None:
            if self.grabbing:
                self.cam.MV_CC_StopGrabbing()
                self.grabbing = False
            if self.opened:
                self.cam.MV_CC_CloseDevice()
                self.opened = False
            if self.handle_created:
                self.cam.MV_CC_DestroyHandle()
                self.handle_created = False
            self.cam = None
        # Do not globally finalize while other camera handles may be in use.


def create_source(config_path):
    return MVSCamera(json.loads(Path(config_path).read_text(encoding='utf-8')))
