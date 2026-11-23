#!/usr/bin/env python3
"""
Hand-eye calibration (กล้อง RealSense -> หุ่น Dobot MG400) ด้วยวัตถุสีเหลือง

ขั้นตอนต่อ 1 จุด (ค่าเริ่มต้น 30 จุด):
  1) เปิดโปรแกรม -> โปรแกรมต่อหุ่นทาง LAN (พอร์ต 29999) แล้ว Enable หุ่นให้เอง
  2) วางวัตถุเหลือง -> กดปุ่มบนแขนหุ่นค้างไว้ ลาก (drag) ปลายหัวจับไปแตะกึ่งกลางผิวบนของวัตถุ
     ปล่อยปุ่ม รอให้หุ่นนิ่ง แล้วกด P (หรือ Enter) ที่หน้าต่างภาพ -> อ่านพิกัดจากหุ่นด้วย GetPose() โดยตรง
  3) ลากหุ่นออกจากมุมกล้อง แล้วกด SPACE -> สแกนตำแหน่งวัตถุในเฟรมกล้อง
  4) ย้ายวัตถุไปตำแหน่ง/ความสูงใหม่ แล้วทำซ้ำ
  ครบทุกจุด -> fit transform, บันทึก handeye_calibration.json แล้ว Disable หุ่นให้เอง
  ปิดโปรแกรมทางไหนก็ตาม (q / Ctrl+C / kill) -> Disable หุ่นก่อนออกเสมอ

ปุ่มที่หน้าต่างภาพ:
  P / Enter = อ่านพิกัดหุ่น   SPACE = สแกน   U = ย้อนกลับ   F = fit ตอนนี้   R = เริ่มใหม่
  E = ClearError + Enable หุ่นใหม่   Q = ออก

หมายเหตุ: ตัวควบคุม MG400 รับ client ต่อพอร์ตได้ทีละตัว -> ห้ามเปิด Project.py / DobotStudio Pro ค้างไว้พร้อมกัน
fit จาก CSV เดิม (ไม่ต้องต่อหุ่น): python3 Handeye_calibration.py --fit calib_points.csv [--model rigid|affine]
"""
import argparse
import atexit
import csv
import json
import os
import re
import signal
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


try:
    import detect_object as _do
except ImportError:
    _do = None


def _local_imgmsg_to_bgr(msg):
    enc = msg.encoding.lower()
    ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1, '8uc1': 1, '8uc3': 3}.get(enc)
    if ch is None:
        raise ValueError(f'ไม่รองรับ encoding ภาพสี: {msg.encoding}')
    buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * ch].reshape(msg.height, msg.width, ch) if ch > 1 else buf[:, :msg.width]
    if enc == 'rgb8':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == 'rgba8':
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if enc == 'bgra8':
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if ch == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _local_imgmsg_to_array(msg):
    """depth -> uint16 mm"""
    enc = msg.encoding.lower()
    if enc in ('16uc1', 'mono16'):
        dt, scale = np.uint16, None
    elif enc == '32fc1':
        dt, scale = np.float32, 1000.0           # เมตร -> mm
    else:
        raise ValueError(f'ไม่รองรับ encoding depth: {msg.encoding}')
    dt = np.dtype(dt).newbyteorder('>' if msg.is_bigendian else '<')
    row = msg.step // dt.itemsize
    arr = np.frombuffer(msg.data, dt).reshape(msg.height, row)[:, :msg.width]
    if scale is None:
        return arr.astype(np.uint16)
    arr = np.nan_to_num(arr.astype(np.float32) * scale, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, 0, 65535).astype(np.uint16)


def _local_compute_crop_rect(iw, ih, x, y, w, h):
    if w <= 0 or h <= 0:
        return 0, 0, iw, ih
    x0 = int(min(max(x, 0), iw - 1))
    y0 = int(min(max(y, 0), ih - 1))
    return x0, y0, int(min(w, iw - x0)), int(min(h, ih - y0))


imgmsg_to_bgr = getattr(_do, 'imgmsg_to_bgr', None) or _local_imgmsg_to_bgr
imgmsg_to_array = getattr(_do, 'imgmsg_to_array', None) or _local_imgmsg_to_array
compute_crop_rect = getattr(_do, 'compute_crop_rect', None) or _local_compute_crop_rect
# depth flat-field: ใช้ได้เฉพาะเมื่อ detect_object.py มีครบ 3 ฟังก์ชัน
load_flat_calibration = getattr(_do, 'load_flat_calibration', None)
flat_slice = getattr(_do, 'flat_slice', None)
apply_flat_correction = getattr(_do, 'apply_flat_correction', None)
HAVE_FLAT = None not in (load_flat_calibration, flat_slice, apply_flat_correction)


# ============================================================================
#  ส่วนที่ 0: Dobot MG400 ผ่าน LAN (พอร์ต Dashboard 29999) : Enable / Disable / GetPose / RobotMode
#  (ไม่สั่งหุ่นเคลื่อนที่ : ผู้ใช้ลากหุ่นเองในโหมด drag)
# ============================================================================
DOBOT_DASH_PORT = 29999
MODE_DISABLED, MODE_ENABLED, MODE_ERROR = 4, 5, 9
MODE_ACTIVE = (5, 6, 7, 8, 10, 11)        # สถานะที่ถือว่าหุ่น Enable อยู่ (6 = drag, 11 = jog)
MODE_NAMES = {1: 'init', 2: 'brake open', 3: 'power off', 4: 'disabled', 5: 'enabled', 6: 'drag',
              7: 'running', 8: 'recording', 9: 'ALARM', 10: 'pause', 11: 'jog'}


class DobotError(RuntimeError):
    pass


class DobotDashboard:
    """client พอร์ต 29999 ของ MG400 : ส่งคำสั่ง 1 ครั้ง อ่านคำตอบ 1 ครั้ง ('ErrorID,{ค่า},คำสั่ง;')"""

    def __init__(self, ip, port=DOBOT_DASH_PORT, log=None):
        self.ip, self.port = ip, int(port)
        self.log = log or (lambda m: print(f'[Dobot] {m}', flush=True))
        self.sock, self.buf, self.last_raw = None, b'', ''
        self.lock = threading.RLock()       # thread อ่านสถานะ กับ thread ปิดโปรแกรม ใช้ socket เดียวกัน

    def _say(self, msg):
        try:
            self.log(msg)
        except Exception:                   # logger ของ ROS อาจใช้ไม่ได้แล้วหลัง rclpy.shutdown()
            print(f'[Dobot] {msg}', flush=True)

    @property
    def connected(self):
        return self.sock is not None

    def open(self, timeout=3.0, retry_s=8.0):
        """ต่อ socket ; หุ่นตอบ 'Connection refused' = ลองใหม่ทุก 0.5 s ไม่เกิน retry_s (หุ่นปล่อยการเชื่อมต่อเก่าช้า)"""
        with self.lock:
            self.close()
            t0 = time.monotonic()
            while True:
                try:
                    self.sock = socket.create_connection((self.ip, self.port), timeout=timeout)
                    break
                except ConnectionRefusedError as e:
                    if time.monotonic() - t0 >= retry_s:
                        raise DobotError(
                            f'ต่อ Dobot {self.ip}:{self.port} ไม่ได้: {e} -> มีโปรแกรมอื่นถือพอร์ตนี้อยู่ '
                            '(Project.py / DobotStudio Pro) หรือหุ่นยังไม่พร้อม') from e
                    time.sleep(0.5)
                except OSError as e:
                    raise DobotError(f'ต่อ Dobot {self.ip}:{self.port} ไม่ได้: {e} '
                                     '(เช็กสาย LAN / IP ของหุ่น / IP เครื่องนี้ต้องอยู่วงเดียวกัน)') from e
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.buf = b''

    def close(self):
        with self.lock:
            if self.sock is not None:
                for fn in (lambda: self.sock.shutdown(socket.SHUT_RDWR), lambda: self.sock.close()):
                    try:
                        fn()
                    except OSError:
                        pass
            self.sock, self.buf = None, b''

    def call(self, cmd, timeout=5.0):
        """คืน (error_id, ข้อความในปีกกา) ; ต่อไม่ได้/หมดเวลา = DobotError"""
        with self.lock:
            if self.sock is None:
                raise DobotError('ยังไม่ได้ต่อ Dobot')
            try:
                self.sock.settimeout(timeout)
                self.sock.sendall(cmd.encode('ascii'))
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise ConnectionError('หุ่นปิดการเชื่อมต่อ')
                self.buf += chunk
                self.sock.settimeout(0.1)           # คำตอบที่ไม่มี ';' : ถือว่าจบเมื่อเงียบ 0.1 s
                while b';' not in self.buf:
                    try:
                        chunk = self.sock.recv(4096)
                    except socket.timeout:
                        break
                    if not chunk:
                        raise ConnectionError('หุ่นปิดการเชื่อมต่อ')
                    self.buf += chunk
            except OSError as e:
                self.close()
                raise DobotError(f'สื่อสารกับ Dobot ไม่สำเร็จขณะส่ง {cmd}: {e}') from e
            if b';' in self.buf:
                reply, _, self.buf = self.buf.partition(b';')
            else:
                reply, self.buf = self.buf, b''
        text = reply.decode('ascii', 'replace').strip()
        self.last_raw = text
        m = re.match(r'^\s*(-?\d+)\s*,\s*\{(.*)\}', text)
        if m:
            return int(m.group(1)), m.group(2)
        m = re.match(r'^\s*\{(.*?)\}', text)
        if m:
            return 0, m.group(1)
        if re.fullmatch(r'\s*-?\d+(\.\d+)?\s*', text):
            return 0, text.strip()
        name = cmd.split('(')[0].strip()           # firmware บางรุ่นตอบแค่ชื่อคำสั่งกลับมา = สำเร็จ
        if name and re.fullmatch(rf'\s*{re.escape(name)}\s*\(.*\)\s*', text):
            return 0, ''
        raise DobotError(f'คำตอบจากหุ่นอ่านไม่ออก: {text!r} (คำสั่ง {cmd})')

    def ok(self, cmd, timeout=5.0):
        err, val = self.call(cmd, timeout)
        if err != 0:
            raise DobotError(f'หุ่นปฏิเสธคำสั่ง {cmd} (ErrorID {err}) raw={self.last_raw!r}')
        return val

    def robot_mode(self):
        return int(self.ok('RobotMode()').split(',')[0])

    def pose(self):
        """(x, y, z, r) ของ TCP ในพิกัด Cartesian ของหุ่น (mm, องศา)"""
        v = [float(s) for s in self.ok('GetPose()').replace('[', '').replace(']', '').split(',') if s.strip()]
        if len(v) < 4:
            raise DobotError(f'GetPose() ตอบมาไม่ครบ: {v}')
        return tuple(v[:4])

    def _send_tolerant(self, cmd, timeout=10.0):
        try:
            self.ok(cmd, timeout)
        except DobotError as e:
            if 'อ่านไม่ออก' not in str(e):      # คำตอบรูปแบบแปลก: ให้ RobotMode เป็นตัวตัดสิน
                raise

    def enable(self, timeout_s=15.0):
        """ClearError (ถ้า alarm) + EnableRobot + รอจนหุ่นพร้อม"""
        if self.robot_mode() == MODE_ERROR:
            self._say('หุ่นอยู่ในสถานะ alarm -> ClearError')
            self.ok('ClearError()')
            time.sleep(0.3)
        self._send_tolerant('EnableRobot()')
        t0 = time.monotonic()
        while (mode := self.robot_mode()) not in MODE_ACTIVE:
            if time.monotonic() - t0 > timeout_s:
                raise DobotError(f'หุ่นไม่พร้อม (RobotMode={mode}) ภายใน {timeout_s:g} s -> เช็กปุ่ม E-Stop / alarm')
            time.sleep(0.2)
        return mode

    def disable(self, wait_s=3.0, tries=2):
        """DisableRobot แล้วตรวจ RobotMode จริง (ยัง Enable = ส่งซ้ำ) ; คืน True เมื่อหุ่นออกจากสถานะ Enable แล้ว"""
        if self.robot_mode() not in MODE_ACTIVE:
            return True
        for n in range(1, tries + 1):
            self._send_tolerant('DisableRobot()', 5.0)
            t0 = time.monotonic()
            while time.monotonic() - t0 < wait_s:
                if self.robot_mode() not in MODE_ACTIVE:
                    return True
                time.sleep(0.2)
            self._say(f'ส่ง DisableRobot ครั้งที่ {n} แล้วหุ่นยังเป็น Enable')
        return False


# ============================================================================
#  ส่วนที่ 1: ตำแหน่งวัตถุสีเหลืองเทียบเฟรมกล้อง
# ============================================================================
def detect_yellow_objects(bgr, depth_mm, hsv_lo=(20, 100, 120), hsv_hi=(35, 255, 255), min_area_px=300):
    """
    หา blob สีเหลืองด้วย HSV คืน (list ของ dict เรียงใหญ่ -> เล็ก, mask รวม)
    dict: bbox, area_px, touches_edge, z_mm (median depth ของผิวบน), centroid (u, v) บนภาพ crop
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    H, W = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
        x, y, w, h = cv2.boundingRect(c)
        obj = np.zeros((H, W), np.uint8)
        cv2.drawContours(obj, [c], -1, 255, -1)
        m = cv2.moments(obj, binaryImage=True)
        if m['m00'] <= 0:
            continue
        # depth: ใช้ส่วนในของวัตถุ (erode ตัดขอบที่ depth มักเพี้ยน)
        inner = cv2.erode(obj, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        if cv2.countNonZero(inner) < 20:
            inner = obj
        d = depth_mm[inner > 0]
        d = d[d > 0]
        z = float(np.median(d)) if d.size >= 10 else None
        results.append({
            'bbox': (x, y, w, h), 'area_px': float(area), 'z_mm': z,
            'centroid': (m['m10'] / m['m00'], m['m01'] / m['m00']),
            'touches_edge': x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1,
        })
    results.sort(key=lambda o: o['area_px'], reverse=True)
    return results, mask


def find_yellow_positions(bgr, depth_mm, intr, crop_offset=(0, 0),
                          hsv_lo=(20, 100, 120), hsv_hi=(35, 255, 255), min_area_px=300):
    """
    หาวัตถุสีเหลืองทุกชิ้น (ใหญ่ -> เล็ก) แล้วคืนตำแหน่ง 3 มิติเทียบเฟรมกล้อง (optical, mm)
      bgr, depth_mm : ภาพสี / depth (uint16 mm) ที่ crop แล้ว ขนาดเท่ากัน (aligned)
      intr          : (fx, fy, cx, cy) ของภาพเต็ม (จาก camera_info)
      crop_offset   : (crop_x0, crop_y0) มุมซ้ายบนของกรอบ crop บนภาพเต็ม
    คืน list ของ dict: xyz_mm (np.array [X, Y, Z]), uv_crop (centroid บนภาพ crop),
                       bbox, area_px, touches_edge
    """
    if intr is None:
        return []
    fx, fy, cx, cy = intr
    ox, oy = crop_offset
    results, _ = detect_yellow_objects(bgr, depth_mm, hsv_lo, hsv_hi, min_area_px)
    out = []
    for o in results:
        if o['z_mm'] is None:
            continue
        u, v = o['centroid']
        z = o['z_mm']
        out.append({
            'xyz_mm': np.array([(u + ox - cx) * z / fx, (v + oy - cy) * z / fy, z]),
            'uv_crop': (u, v), 'bbox': o['bbox'], 'area_px': o['area_px'],
            'touches_edge': o['touches_edge'],
        })
    return out


# ============================================================================
#  ส่วนที่ 2: fit transform กล้อง -> หุ่น  (numpy ล้วน)
# ============================================================================
def fit_rigid(P, Q):
    """Kabsch/SVD: หา R, t ที่ทำให้ R @ p + t ใกล้ q ที่สุด คืนเมทริกซ์ 3x4 [R | t]"""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    d = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0 else -1.0     # กันได้ reflection
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return np.hstack([R, (cq - R @ cp)[:, None]])


def fit_affine(P, Q):
    """least squares แบบ affine 12 พารามิเตอร์ คืนเมทริกซ์ 3x4"""
    X, *_ = np.linalg.lstsq(np.hstack([P, np.ones((len(P), 1))]), Q, rcond=None)
    return X.T


def apply_transform(M, P):
    P = np.atleast_2d(np.asarray(P, np.float64))
    return P @ M[:, :3].T + M[:, 3]


def planarity(P):
    """สัดส่วน singular value เล็กสุด/ใหญ่สุด ของจุด (ใกล้ 0 = จุดอยู่ระนาบเดียวกัน)"""
    sv = np.linalg.svd(P - P.mean(0), compute_uv=False)
    return float(sv[2] / max(sv[0], 1e-9))


def calibrate(P, Q, model='rigid'):
    """
    fit transform จากจุดคู่ P (กล้อง) -> Q (หุ่น)  Nx3 mm
    คืน dict: M (3x4), model, residuals (mm ต่อจุด), rmse, loo (leave-one-out error ต่อจุด), notes
    """
    P, Q = np.asarray(P, np.float64), np.asarray(Q, np.float64)
    n = len(P)
    if n < 4:
        raise ValueError('ต้องมีอย่างน้อย 4 จุด')
    notes = []
    flat = planarity(P)
    if flat < 0.02:
        notes.append('จุดแทบอยู่ระนาบเดียวกัน (ความสูงไม่หลากหลาย) ควรเพิ่มจุดที่ความสูงต่างกัน')
    if model == 'affine' and (n < 6 or flat < 0.02):
        notes.append('affine ต้องมี >= 6 จุดและไม่อยู่ระนาบเดียวกัน จึงใช้ rigid แทน')
        model = 'rigid'
    fit = fit_affine if model == 'affine' else fit_rigid
    M = fit(P, Q)
    res = np.linalg.norm(apply_transform(M, P) - Q, axis=1)
    loo = None
    if n >= 5:
        loo = np.zeros(n)
        for i in range(n):
            keep = np.arange(n) != i
            loo[i] = np.linalg.norm(apply_transform(fit(P[keep], Q[keep]), P[i]) - Q[i])
    return {'M': M, 'model': model, 'residuals': res, 'rmse': float(np.sqrt(np.mean(res ** 2))),
            'loo': loo, 'planarity': flat, 'notes': notes}


def format_report(P, Q, r):
    lines = [f"calibrate ({r['model']}) จาก {len(P)} จุด",
             ' #   cam X,Y,Z (mm)               robot X,Y,Z (mm)             residual   LOO']
    for i in range(len(P)):
        loo = f"{r['loo'][i]:6.2f}" if r['loo'] is not None else '   -  '
        lines.append(
            f"{i + 1:2d}   ({P[i, 0]:7.1f},{P[i, 1]:7.1f},{P[i, 2]:7.1f})   "
            f"({Q[i, 0]:7.1f},{Q[i, 1]:7.1f},{Q[i, 2]:7.1f})   {r['residuals'][i]:6.2f}   {loo}")
    s = f"RMSE = {r['rmse']:.2f} mm   max residual = {r['residuals'].max():.2f} mm"
    if r['loo'] is not None:
        s += f"   LOO mean = {r['loo'].mean():.2f} mm  max = {r['loo'].max():.2f} mm"
    lines.append(s)
    lines += [f'หมายเหตุ: {t}' for t in r['notes']]
    if r['rmse'] > 3.0:
        lines.append('หมายเหตุ: RMSE > 3 mm  ตรวจว่าแตะหุ่นที่จุดเดียวกับที่กล้องวัด (กึ่งกลางผิวบน), '
                     'crop/depth flat ตรงกับตอนใช้งาน, และไม่มีจุดที่อ่านค่าผิด (ดูจุดที่ residual/LOO สูง)')
    return '\n'.join(lines)


class CameraToRobot:
    """transform พิกัดกล้อง (optical, mm) -> พิกัดหุ่น (mm)  p_robot = M[:, :3] @ p_cam + M[:, 3]"""

    def __init__(self, M, meta=None):
        self.M = np.asarray(M, np.float64).reshape(3, 4)
        self.meta = meta or {}

    def cam_to_robot(self, p_cam):
        p = apply_transform(self.M, p_cam)
        return tuple(float(v) for v in p[0]) if np.ndim(p_cam) == 1 else p

    def save(self, path, P, Q, r, extra=None):
        data = {
            'created': datetime.now().isoformat(timespec='seconds'),
            'camera_frame': 'RealSense optical frame (X right, Y down, Z forward), mm',
            'model': r['model'], 'M': self.M.tolist(),
            'R': self.M[:, :3].tolist(), 't': self.M[:, 3].tolist(),
            'n_points': int(len(P)), 'rmse_mm': r['rmse'],
            'max_residual_mm': float(r['residuals'].max()),
            'loo_mean_mm': None if r['loo'] is None else float(r['loo'].mean()),
            'loo_max_mm': None if r['loo'] is None else float(r['loo'].max()),
            'points': [{'cam': P[i].tolist(), 'robot': Q[i].tolist(),
                        'residual_mm': float(r['residuals'][i])} for i in range(len(P))],
        }
        data.update(extra or {})
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        return cls(d['M'], meta=d)


_CSV_HEADER = ['cam_x', 'cam_y', 'cam_z', 'std_x', 'std_y', 'std_z', 'rob_x', 'rob_y', 'rob_z', 'rob_r']


def load_pairs_csv(path):
    P, Q = [], []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            P.append([float(row['cam_x']), float(row['cam_y']), float(row['cam_z'])])
            Q.append([float(row['rob_x']), float(row['rob_y']), float(row['rob_z'])])
    return np.array(P), np.array(Q)


# ============================================================================
#  ROS node: เก็บจุดคู่ (พิกัดหุ่นอ่านจากหุ่นโดยตรง) -> fit -> บันทึก
# ============================================================================
class YellowCalibNode(Node):
    def __init__(self):
        super().__init__('yellow_robot_calibration')
        d = self.declare_parameter
        d('color_topic', '/camera/camera/color/image_raw')
        d('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        d('info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        d('cam_topic', '/yellow/position_cam')
        d('display_scale', 3.0)
        d('crop_x', 487)                    # เลื่อนไปขวาให้พ้นขอบโต๊ะไม้ทางซ้าย (ตรงกับ crop ของ Project.py)
        d('crop_y', 310)
        d('crop_width', 280)                # ตัดขอบขวาเข้ามา 20 px ไม่ให้เห็นขอบโต๊ะ
        d('crop_height', 208)
        d('depth_flat_file', '')
        for k, v in (('h_low', 20), ('h_high', 35), ('s_low', 100), ('s_high', 255),
                     ('v_low', 120), ('v_high', 255)):
            d('yellow_' + k, v)
        d('yellow_min_area_px', 300)
        d('num_points', 30)
        d('samples', 30)
        d('max_std_mm', 3.0)
        d('model', 'rigid')
        d('output_dir', '.')
        # ---- หุ่น Dobot MG400 (LAN)
        d('dobot_ip', '192.168.1.6')
        d('dobot_enable_timeout_s', 15.0)
        d('pose_hz', 5.0)                   # ความถี่อ่าน GetPose / RobotMode
        d('still_mm', 0.5)                  # หุ่นต้องนิ่ง (ค่ากระจายไม่เกินนี้) ตอนกด P
        d('still_s', 0.6)                   # ... ต่อเนื่องอย่างน้อยกี่วินาที
        d('clear_xy_mm', 60.0)              # ก่อนสแกน หุ่นต้องห่างจุดที่แตะ (แนว x,y) อย่างน้อยเท่านี้
        d('disable_when_done', True)        # ครบทุกจุด -> Disable หุ่นเอง

        g = lambda n: self.get_parameter(n).get_parameter_value()  # noqa: E731
        self.scale = g('display_scale').double_value
        self.crop = (g('crop_x').integer_value, g('crop_y').integer_value,
                     g('crop_width').integer_value, g('crop_height').integer_value)
        self.hsv_lo = (g('yellow_h_low').integer_value, g('yellow_s_low').integer_value,
                       g('yellow_v_low').integer_value)
        self.hsv_hi = (g('yellow_h_high').integer_value, g('yellow_s_high').integer_value,
                       g('yellow_v_high').integer_value)
        self.min_area = g('yellow_min_area_px').integer_value
        self.num_points = g('num_points').integer_value
        self.n_samples = g('samples').integer_value
        self.max_std = g('max_std_mm').double_value
        self.model = g('model').string_value.lower()
        self.out_dir = os.path.abspath(g('output_dir').string_value)
        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(self.out_dir, 'calib_points.csv')
        self.json_path = os.path.join(self.out_dir, 'handeye_calibration.json')

        self.dobot_ip = g('dobot_ip').string_value
        self.enable_timeout = g('dobot_enable_timeout_s').double_value
        self.pose_period = 1.0 / max(g('pose_hz').double_value, 0.5)
        self.still_mm = g('still_mm').double_value
        self.still_s = max(g('still_s').double_value, 0.2)
        self.clear_xy_mm = g('clear_xy_mm').double_value
        self.disable_when_done = g('disable_when_done').bool_value

        self.flat = None
        self._flat_warned = False
        fpath = g('depth_flat_file').string_value
        if fpath:
            if HAVE_FLAT:
                self.flat = load_flat_calibration(fpath)
            else:
                self.get_logger().warn('detect_object.py ไม่มีฟังก์ชัน depth flat จึงไม่ใช้ depth_flat_file')

        self.intrinsics = None
        self.lock = threading.Lock()
        self.latest = None            # (color, depth, header, crop_x0, crop_y0)
        self.pairs = []               # list ของ (cam xyz, std xyz, robot xyz, robot r)
        self.T = None                 # CameraToRobot หลัง fit
        self.state = 'await_robot'    # await_robot -> ready_scan -> capturing -> (จุดถัดไป) ... -> done
        self.robot_pending = None     # พิกัดหุ่นของจุดปัจจุบัน (อ่านจากหุ่นก่อนสแกน)
        self.robot_pending_r = 0.0
        self.target = None            # วัตถุเหลืองที่เห็นในเฟรมล่าสุด
        self.samples = []
        self.cap_t0 = 0.0
        self.last_stamp = None
        self.msg = ''
        self.msg_until = 0.0

        # ---- หุ่น: thread เดียวเป็นเจ้าของ socket (Enable / Disable / อ่าน pose ต่อเนื่อง) -> หน้าต่างภาพไม่ค้าง
        self.bot = DobotDashboard(self.dobot_ip, log=self._robot_log)
        self.robot_state = 'connecting'   # connecting | enabling | enabled | disabling | disabled | alarm | error
        self.robot_mode = None
        self.pose_hist = deque(maxlen=60)  # (t, xyz np.array, r)
        self._req_lock = threading.Lock()
        self._req = 'enable'               # คำขอที่รอทำ: 'enable' | 'disable' | None
        self._want_enabled = True
        self._robot_stop = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False
        self._robot_thread = threading.Thread(target=self._robot_loop, name='dobot', daemon=True)
        self._robot_thread.start()

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.cam_pub = self.create_publisher(PointStamped, g('cam_topic').string_value, 10)
        self.create_subscription(CameraInfo, g('info_topic').string_value, self.on_info, qos)
        cs = Subscriber(self, Image, g('color_topic').string_value, qos_profile=qos)
        ds = Subscriber(self, Image, g('depth_topic').string_value, qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer([cs, ds], queue_size=5, slop=0.05)
        self.sync.registerCallback(self.on_frames)

        self.win = 'Yellow calibration'
        cv2.namedWindow(self.win)
        self.create_timer(1.0 / 30.0, self.on_display)
        self.get_logger().info(
            f'พร้อม calibrate {self.num_points} จุด (model={self.model}, หุ่น {self.dobot_ip}) | '
            'ขั้นตอน: วางวัตถุเหลือง -> ลากหุ่น (กดปุ่มบนแขนค้าง) ไปแตะกึ่งกลางผิวบน -> ปล่อยให้นิ่ง กด P '
            '-> ลากหุ่นออกจากมุมกล้อง -> กด SPACE เพื่อสแกน')
        self.get_logger().info(f'จุดที่เก็บจะถูกบันทึกที่ {self.csv_path}')
        self.prompt_robot()

    # ---------------------------------------------------------------- หุ่น (thread แยก)
    def _robot_log(self, msg):
        try:
            self.get_logger().info(msg)
        except Exception:
            print(f'[Dobot] {msg}', flush=True)

    def request_robot(self, kind):
        """ขอให้ thread หุ่น 'enable' / 'disable' (ทำในรอบถัดไป)"""
        with self._req_lock:
            self._req = kind
            self._want_enabled = kind == 'enable'

    def _take_req(self):
        with self._req_lock:
            r, self._req = self._req, None
            return r

    def _robot_loop(self):
        last_err = None
        while not self._robot_stop.is_set():
            t0 = time.monotonic()
            req = None
            try:
                if not self.bot.connected:
                    self.robot_state = 'connecting'
                    self.bot.open()
                    self.get_logger().info(f'ต่อ Dobot {self.dobot_ip}:{DOBOT_DASH_PORT} ได้แล้ว')
                last_err = None
                req = self._take_req()
                if req == 'enable':
                    self.robot_state = 'enabling'
                    self.say('กำลัง Enable หุ่น...', en='enabling robot...')
                    mode = self.bot.enable(self.enable_timeout)
                    self.say(f'Enable หุ่นแล้ว (RobotMode {mode}): กดปุ่มบนแขนหุ่นค้างไว้เพื่อลากหุ่น',
                             en='robot enabled - hold the arm button to drag')
                elif req == 'disable':
                    self.robot_state = 'disabling'
                    if self.bot.disable():
                        self.say('Disable หุ่นแล้ว', en='robot disabled')
                    else:
                        self.say('Disable หุ่นไม่สำเร็จ: กด Disable ที่หุ่น/DobotStudio เอง', True,
                                 en='robot disable FAILED')
                x, y, z, r = self.bot.pose()
                mode = self.bot.robot_mode()
                now = time.monotonic()
                self.pose_hist.append((now, np.array([x, y, z], np.float64), float(r)))
                self.robot_mode = mode
                new = 'enabled' if mode in MODE_ACTIVE else 'alarm' if mode == MODE_ERROR else 'disabled'
                if new != self.robot_state and self.robot_state not in ('enabling', 'disabling', 'connecting'):
                    if new == 'alarm':
                        self.say('หุ่นเข้าสถานะ alarm: ตรวจหุ่นแล้วกด E เพื่อ ClearError + Enable ใหม่', True,
                                 en='robot ALARM - press E to re-enable')
                    elif new == 'disabled' and self._want_enabled:
                        self.say('หุ่นไม่ได้ Enable อยู่ (E-Stop?): กด E เพื่อ Enable ใหม่', True,
                                 en='robot not enabled - press E')
                self.robot_state = new
            except DobotError as e:
                msg = str(e)
                if msg != last_err:
                    self.say(f'Dobot: {msg}', True, en='robot error - see terminal')
                    last_err = msg
                if not self.bot.connected:
                    if req is not None:                  # หลุดระหว่างทำคำขอ: ทำใหม่เมื่อต่อได้
                        with self._req_lock:
                            if self._req is None:
                                self._req = req
                    self._robot_stop.wait(1.0)
                self.robot_state = 'error'
            self._robot_stop.wait(max(0.0, self.pose_period - (time.monotonic() - t0)))

    def live_pose(self, max_age=1.0):
        """(xyz, r) ล่าสุดจากหุ่น หรือ None ถ้าเก่าเกิน max_age วินาที"""
        if not self.pose_hist:
            return None
        t, xyz, r = self.pose_hist[-1]
        return (xyz, r) if time.monotonic() - t <= max_age else None

    def shutdown_robot(self):
        """ปิดโปรแกรม: หยุด thread หุ่น -> DisableRobot -> ปิด socket (เรียกซ้ำได้ ทำจริงครั้งเดียว)"""
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
        self._robot_stop.set()
        self._robot_thread.join(timeout=3.0)
        try:
            if not self.bot.connected:
                self.bot.open(timeout=2.0, retry_s=2.0)
            ok = self.bot.disable()
            print(f"[Dobot] ปิดโปรแกรม: {'Disable หุ่นแล้ว' if ok else 'Disable หุ่นไม่สำเร็จ'}", flush=True)
        except Exception as e:
            print(f'[Dobot] ปิดโปรแกรม: Disable หุ่นไม่สมบูรณ์: {type(e).__name__}: {e}', flush=True)
        finally:
            self.bot.close()

    # ---------------------------------------------------------------- กล้อง
    def on_info(self, msg):
        if self.intrinsics is None and msg.k[0] > 0 and msg.k[4] > 0:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.get_logger().info(
                f'intrinsics: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} cx={msg.k[2]:.1f} cy={msg.k[5]:.1f}')

    def on_frames(self, color_msg, depth_msg):
        try:
            color, depth = imgmsg_to_bgr(color_msg), imgmsg_to_array(depth_msg)
        except Exception as e:
            self.get_logger().error(f'แปลงภาพไม่สำเร็จ: {e}')
            return
        ih, iw = color.shape[:2]
        x0, y0, cw, ch = compute_crop_rect(iw, ih, *self.crop)
        color = np.ascontiguousarray(color[y0:y0 + ch, x0:x0 + cw])
        depth = np.ascontiguousarray(depth[y0:y0 + ch, x0:x0 + cw])
        if self.flat is not None:
            delta = flat_slice(self.flat, (x0, y0, cw, ch), (iw, ih))
            if delta is not None:
                depth = apply_flat_correction(depth, delta)
            elif not self._flat_warned:
                self._flat_warned = True
                self.get_logger().warn('depth flat: กรอบ crop/ขนาดภาพไม่ตรงกับตอน calibrate จึงไม่แก้ค่า')
        with self.lock:
            self.latest = (color, depth, depth_msg.header, x0, y0)

    # ---------------------------------------------------------------- state machine
    def say(self, text, warn=False, en=''):
        """text = ข้อความเต็มในเทอร์มินัล, en = ข้อความสั้นภาษาอังกฤษบนหน้าต่างภาพ (putText แสดงภาษาไทยไม่ได้)"""
        try:
            if warn:
                self.get_logger().warn(text)
            else:
                self.get_logger().info(text)
        except Exception:
            print(text, flush=True)
        self.msg, self.msg_until = en, time.monotonic() + 4.0

    def prompt_robot(self):
        """เริ่มจุดใหม่: รอให้ลากหุ่นไปแตะวัตถุแล้วกด P"""
        self.robot_pending = None
        idx = len(self.pairs)
        if idx >= self.num_points:
            return
        self.state = 'await_robot'
        self.get_logger().info(
            f'[จุด {idx + 1}/{self.num_points}] วางวัตถุเหลืองตำแหน่ง/ความสูงใหม่ -> ลากหุ่นไปแตะกึ่งกลางผิวบน '
            '-> ปล่อยปุ่มให้หุ่นนิ่ง แล้วกด P (หรือ Enter) ที่หน้าต่างภาพ')

    def record_pose(self):
        """กด P: อ่านพิกัดหุ่นจาก GetPose (เฉลี่ยช่วงที่หุ่นนิ่ง still_s วินาที)"""
        if self.state == 'ready_scan':
            self.say('มีพิกัดหุ่นของจุดนี้แล้ว: กด U เพื่ออ่านใหม่ หรือลากหุ่นออกแล้วกด SPACE',
                     True, en='robot pose already recorded - U to redo')
            return
        if self.state != 'await_robot':
            return
        if self.robot_mode == MODE_ERROR:
            self.say('หุ่นอยู่ในสถานะ alarm: กด E เพื่อ ClearError + Enable ก่อน', True, en='robot ALARM - press E')
            return
        if self.live_pose() is None:
            self.say(f'อ่านตำแหน่งหุ่นไม่ได้ (ยังต่อ Dobot {self.dobot_ip} ไม่ได้)', True, en='no robot pose (not connected)')
            return
        now = time.monotonic()
        win = [h for h in list(self.pose_hist) if now - h[0] <= self.still_s + 0.05]
        if len(win) < 3 or win[-1][0] - win[0][0] < 0.6 * self.still_s:
            self.say('ข้อมูลตำแหน่งยังไม่พอ: รอให้หุ่นนิ่งสักครู่แล้วกด P ใหม่', True, en='hold still and press P again')
            return
        P = np.array([h[1] for h in win])
        xyz = P.mean(0)
        dev = float(np.linalg.norm(P - xyz, axis=1).max())
        if dev > self.still_mm:
            self.say(f'หุ่นยังขยับ (แกว่ง {dev:.2f} mm > {self.still_mm:g}): ปล่อยปุ่มให้หุ่นนิ่งแล้วกด P ใหม่',
                     True, en=f'robot moving ({dev:.1f} mm) - press P again')
            return
        self.robot_pending, self.robot_pending_r = xyz, float(win[-1][2])
        self.state = 'ready_scan'
        self.say(f'จุด {len(self.pairs) + 1}: หุ่น ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) R {self.robot_pending_r:.2f} '
                 '-> ลากหุ่นออกจากมุมกล้อง แล้วกด SPACE เพื่อสแกน', en='robot pose recorded - move robot away, SPACE')

    def start_capture(self):
        if self.state == 'await_robot':
            self.say('ลากหุ่นไปแตะวัตถุแล้วกด P ก่อน แล้วค่อยกด SPACE', True, en='record robot pose (P) first')
            return
        if self.state != 'ready_scan':
            return
        if self.intrinsics is None:
            self.say('ยังไม่ได้รับ camera_info รอสักครู่', True, en='waiting for camera_info')
            return
        live = self.live_pose()
        if live is None:
            self.say('อ่านตำแหน่งหุ่นไม่ได้ จึงตรวจไม่ได้ว่าหุ่นออกจากมุมกล้องแล้ว', True, en='no robot pose - cannot scan')
            return
        dxy = float(np.hypot(*(live[0][:2] - self.robot_pending[:2])))
        if dxy < self.clear_xy_mm:
            self.say(f'หุ่นยังอยู่เหนือวัตถุ (ห่าง {dxy:.0f} mm < {self.clear_xy_mm:g}): ลากหุ่นออกจากมุมกล้องก่อน',
                     True, en=f'robot still over object ({dxy:.0f} mm) - move it away')
            return
        if self.target is None:
            self.say('ไม่เห็นวัตถุเหลืองเต็มชิ้น (อย่าให้ชิดขอบภาพ/หุ่นบัง) แล้วกด SPACE ใหม่', True,
                     en='no full yellow object visible - press SPACE again')
            return
        self.samples, self.cap_t0, self.last_stamp = [], time.monotonic(), None
        self.state = 'capturing'
        self.get_logger().info(f'กำลังสแกน {self.n_samples} เฟรม... (อย่าขยับวัตถุ)')

    def feed_capture(self, target, header):
        stamp = (header.stamp.sec, header.stamp.nanosec)
        if stamp != self.last_stamp and target is not None:       # นับเฉพาะเฟรมใหม่ที่เห็นวัตถุ
            self.last_stamp = stamp
            self.samples.append(target['xyz_mm'])
        if len(self.samples) >= self.n_samples:
            arr = np.array(self.samples)
            cam, std = np.median(arr, axis=0), arr.std(axis=0)
            if std.max() > self.max_std:
                self.state = 'ready_scan'
                self.say(f'ค่าแกว่งเกินไป (std {std.max():.1f} mm > {self.max_std}) วัตถุขยับ/depth noise: '
                         'กด SPACE สแกนใหม่', True, en=f'unstable (std {std.max():.1f} mm) - press SPACE again')
                return
            self.get_logger().info(
                f'จุดที่ {len(self.pairs) + 1}: กล้อง X,Y,Z = ({cam[0]:.1f}, {cam[1]:.1f}, {cam[2]:.1f}) mm '
                f'(std {std.max():.2f} mm)')
            self.finish_point(cam, std)
        elif time.monotonic() - self.cap_t0 > max(10.0, self.n_samples / 5.0):
            self.state = 'ready_scan'
            self.say('สแกนไม่สำเร็จ: ไม่เห็นวัตถุเหลืองเต็มชิ้น/ไม่มีเฟรมใหม่ กด SPACE ลองใหม่', True,
                     en='scan failed - press SPACE again')

    def finish_point(self, cam, std):
        rob, rr = self.robot_pending, self.robot_pending_r
        self.pairs.append((cam, std, rob, rr))
        self.robot_pending = None
        self.write_csv()
        self.say(f'บันทึกจุดที่ {len(self.pairs)}/{self.num_points}: '
                 f'กล้อง ({cam[0]:.1f}, {cam[1]:.1f}, {cam[2]:.1f}) -> '
                 f'หุ่น ({rob[0]:.1f}, {rob[1]:.1f}, {rob[2]:.1f})', en=f'point {len(self.pairs)} saved')
        if len(self.pairs) >= self.num_points:
            self.do_fit()
        else:
            self.prompt_robot()

    def write_csv(self):
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(_CSV_HEADER)
            for cam, std, rob, rr in self.pairs:
                w.writerow([f'{v:.3f}' for v in (*cam, *std, *rob, rr)])

    def do_fit(self):
        if len(self.pairs) < 4:
            self.say('ต้องมีอย่างน้อย 4 จุดถึงจะ fit ได้', True, en='need >= 4 points to fit')
            return
        P = np.array([p[0] for p in self.pairs])
        Q = np.array([p[2] for p in self.pairs])
        r = calibrate(P, Q, self.model)
        self.get_logger().info('\n' + format_report(P, Q, r))
        self.T = CameraToRobot(r['M'])
        extra = {'crop_xywh': list(self.crop), 'intrinsics_fx_fy_cx_cy': self.intrinsics,
                 'yellow_hsv_lo': list(self.hsv_lo), 'yellow_hsv_hi': list(self.hsv_hi),
                 'robot_source': f'Dobot MG400 GetPose() @ {self.dobot_ip}'}
        self.T.save(self.json_path, P, Q, r, extra)
        self.say(f'calibrate เสร็จ: RMSE {r["rmse"]:.2f} mm  -> {self.json_path}', en=f'calibrated: RMSE {r["rmse"]:.2f} mm')
        if len(self.pairs) >= self.num_points:
            self.state = 'done'
            if self.disable_when_done:
                self.request_robot('disable')
                self.get_logger().info('ครบทุกจุดแล้ว -> Disable หุ่น (กด U เพื่อแก้จุดล่าสุด / E เพื่อ Enable ใหม่)')

    def undo(self):
        if self.state == 'capturing':
            self.state = 'ready_scan'
            self.say('ยกเลิกการสแกน (พิกัดหุ่นยังอยู่ กด SPACE สแกนใหม่ได้)', en='scan cancelled')
        elif self.state == 'ready_scan':
            self.say('ทิ้งพิกัดหุ่นของจุดนี้: ลากหุ่นไปแตะวัตถุแล้วกด P ใหม่', en='robot pose cleared - press P again')
            self.prompt_robot()
        elif self.pairs:
            was_done = self.state == 'done'
            self.pairs.pop()
            self.write_csv()
            self.T = None
            self.say(f'ลบจุดล่าสุดแล้ว เหลือ {len(self.pairs)} จุด', en=f'last point removed ({len(self.pairs)} left)')
            if was_done:
                self.request_robot('enable')
            self.prompt_robot()

    def reset(self):
        was_done = self.state == 'done'
        self.pairs, self.T = [], None
        self.write_csv()
        self.say('เริ่มใหม่ทั้งหมด', en='reset')
        if was_done or not self._want_enabled:
            self.request_robot('enable')
        self.prompt_robot()

    # ---------------------------------------------------------------- display loop
    def on_display(self):
        with self.lock:
            fr = self.latest
        if fr is None:
            self.handle_key(cv2.waitKey(1) & 0xFF)
            return
        color, depth, header, ox, oy = fr
        dets = find_yellow_positions(color, depth, self.intrinsics, (ox, oy),
                                     self.hsv_lo, self.hsv_hi, self.min_area)
        full = [dd for dd in dets if not dd['touches_edge']]     # ชิ้นที่เห็นเต็ม (ไม่ชิดขอบ)
        target = full[0] if full else None
        self.target = target

        if target is not None:
            p = PointStamped()
            p.header = header
            p.point.x, p.point.y, p.point.z = (float(v) * 0.001 for v in target['xyz_mm'])
            self.cam_pub.publish(p)

        if self.state == 'capturing':
            self.feed_capture(target, header)

        self.draw(color, dets, target)
        self.handle_key(cv2.waitKey(1) & 0xFF)

    def handle_key(self, key):
        if key == ord(' '):
            self.start_capture()
        elif key in (ord('p'), ord('P'), 13, 10):
            self.record_pose()
        elif key in (ord('u'), ord('U')):
            self.undo()
        elif key in (ord('f'), ord('F')):
            self.do_fit()
        elif key in (ord('r'), ord('R')):
            self.reset()
        elif key in (ord('e'), ord('E')):
            self.request_robot('enable')
        elif key in (ord('q'), ord('Q')):
            self.get_logger().info('ออกจากโปรแกรม -> Disable หุ่น...')
            self.shutdown_robot()
            rclpy.shutdown()

    def draw(self, color, dets, target):
        sc = self.scale if self.scale > 0 else 1.0
        img = cv2.resize(color, None, fx=sc, fy=sc,
                         interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)

        def text(t, org, col=(0, 255, 255), size=0.6):
            cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, size, col, 1, cv2.LINE_AA)

        for dd in dets:
            x, y, w, h = (int(v * sc) for v in dd['bbox'])
            good = dd is target
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255) if good else (0, 0, 255), 2)
            u, v = int(dd['uv_crop'][0] * sc), int(dd['uv_crop'][1] * sc)
            cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            X, Y, Z = dd['xyz_mm']
            lines = [f'cam {X:.1f} {Y:.1f} {Z:.1f} mm']
            if dd['touches_edge']:
                lines.append('touches edge: not used')
            if good and self.T is not None:
                rx, ry, rz = self.T.cam_to_robot(dd['xyz_mm'])
                lines.append(f'robot {rx:.1f} {ry:.1f} {rz:.1f} mm')
            for i, t in enumerate(lines):
                text(t, (x, max(y - 8 - 20 * (len(lines) - 1 - i), 18)))

        status = {'await_robot': 'drag robot tip to top-center, hold still, press P',
                  'ready_scan': 'move robot out of view, press SPACE',
                  'capturing': f'scanning {len(self.samples)}/{self.n_samples}',
                  'done': 'calibrated - verify with a new position'}[self.state]
        if self.intrinsics is None:
            status = 'waiting for camera_info'
        text(f'point {min(len(self.pairs) + 1, self.num_points)}/{self.num_points} | {status}',
             (6, 20), (0, 255, 0), 0.6)

        live = self.live_pose()
        rs = self.robot_state
        rcol = {'enabled': (0, 255, 0), 'alarm': (0, 0, 255), 'error': (0, 0, 255)}.get(rs, (0, 200, 255))
        mode_txt = '' if self.robot_mode is None else f' mode {self.robot_mode} ({MODE_NAMES.get(self.robot_mode, "?")})'
        pose_txt = f' | now {live[0][0]:.1f} {live[0][1]:.1f} {live[0][2]:.1f}' if live is not None else ' | no pose'
        text(f'robot {self.dobot_ip}: {rs}{mode_txt}{pose_txt}', (6, 42), rcol, 0.5)
        if self.robot_pending is not None and self.state in ('ready_scan', 'capturing'):
            rp = self.robot_pending
            text(f'recorded robot: {rp[0]:.1f} {rp[1]:.1f} {rp[2]:.1f}', (6, 62), (255, 200, 0), 0.5)
        if self.msg and time.monotonic() < self.msg_until:
            text(self.msg[:90], (6, img.shape[0] - 8), (255, 255, 255), 0.5)
        cv2.imshow(self.win, img)


def fit_from_csv(path, model, out):
    P, Q = load_pairs_csv(path)
    r = calibrate(P, Q, model)
    print(format_report(P, Q, r))
    CameraToRobot(r['M']).save(out, P, Q, r)
    print(f'บันทึก {out}')


def main(args=None):
    argv = sys.argv[1:] if args is None else list(args)
    if '--fit' in argv:
        ap = argparse.ArgumentParser()
        ap.add_argument('--fit', required=True, help='CSV ที่ node บันทึกไว้ (calib_points.csv)')
        ap.add_argument('--model', default='rigid', choices=['rigid', 'affine'])
        ap.add_argument('--out', default='handeye_calibration.json')
        a, _ = ap.parse_known_args(argv)
        fit_from_csv(a.fit, a.model, a.out)
        return
    rclpy.init(args=args)
    node = YellowCalibNode()
    atexit.register(node.shutdown_robot)                 # ออกทางไหนก็ Disable หุ่น (เรียกซ้ำได้)

    def _on_term(signum, frame):                         # kill / ปิดเทอร์มินัล -> ออกทางเดียวกับ Ctrl+C
        raise KeyboardInterrupt
    for _sig in (signal.SIGTERM, getattr(signal, 'SIGHUP', None)):
        if _sig is not None:
            signal.signal(_sig, _on_term)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown_robot()                            # DisableRobot ก่อนออกเสมอ
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
