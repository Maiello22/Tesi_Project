#!/usr/bin/env python3
"""Prova offline (senza ROS ne' Gazebo) della logica di inspection_node.
Mondo finto: 4 marker (201, 202 raggiungibili; 203 fuori portata; 204 troppo lontano),
camera pinhole 672x376 fx=235, rumore di stima ~ Z^2, braccio che si muove a 0.15 m/s.
Caso 1: CLIK ideale. Caso 2: CLIK con errore sistematico di 7 mm / 1 grado e marker 202
rimosso dopo la mappatura (deve risultare 'perso').
Uso: python3 test/test_inspection_logic.py"""
import sys, math, numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from inspection_node import *

rng = np.random.default_rng(1)
FX, CX, CY = 235.0, 336.0, 188.0

def marker_T(r, az_deg, z):
    """marker a raggio r, azimut az, rivolto verso l'asse Z della base (normale verso il robot)."""
    a = math.radians(az_deg)
    p = np.array([r*math.cos(a), r*math.sin(a), z])
    zM = -np.array([math.cos(a), math.sin(a), 0.0])        # normale verso la base
    yM = np.array([0, 0, 1.0])                              # su
    xM = np.cross(yM, zM)
    return make_T(np.column_stack([xM, yM, zM]), p)

WORLD = {201: marker_T(1.00, -90, 0.75), 202: marker_T(0.90, 30, 0.60),
         203: marker_T(1.30, 150, 0.75), 204: marker_T(2.30, 90, 0.75)}

def cam_from(pos, look):
    z = look/np.linalg.norm(look); y = np.array([0, 0, -1.0]); y = y - z*np.dot(y, z); y /= np.linalg.norm(y)
    return make_T(np.column_stack([np.cross(y, z), y, z]), pos)

class FakeIO:
    def __init__(s):
        s.t = 0.0
        s.cam = cam_from(np.array([0.059, -0.250, 1.126]), np.array([0, -1.0, 0]))   # posa di initial_positions.yaml (0.998 m dalla spalla)
        s.goal = s.cam.copy(); s.logs = []; s.sent = 0; s.rejected = 0; s.min_r = 9
    def now(s): return s.t
    def ready(s): return True
    def cam_pose(s): return s.cam.copy()
    def send_target(s, T):
        s.sent += 1
        if np.linalg.norm(T[:3,3]-[0,0,0.1625]) > 0.95: s.rejected += 1; return
        s.goal = T.copy()
    def save_photo(s, n): return n+".png"
    def write_report(s, rows): s.rows = rows
    def log(s, m): print(f"[{s.t:7.2f}] {m}")
    def state(s, m): print(f"[{s.t:7.2f}] -- {m}")
    def step(s, dt):
        p, g = s.cam[:3,3], s.goal[:3,3]
        d = g-p; n = np.linalg.norm(d)
        p = p + (d if n < 0.15*dt else d/n*0.15*dt)
        Rr = s.cam[:3,:3].T @ s.goal[:3,:3]; ang = rot_angle(Rr)
        if ang > 1e-9:
            w = np.array([Rr[2,1]-Rr[1,2], Rr[0,2]-Rr[2,0], Rr[1,0]-Rr[0,1]])/(2*math.sin(ang))
            a = min(ang, 0.6*dt); K = np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
            R = s.cam[:3,:3] @ (np.eye(3)+math.sin(a)*K+(1-math.cos(a))*K@K)
        else: R = s.cam[:3,:3]
        s.cam = make_T(R, p)
        s.min_r = min(s.min_r, math.hypot(p[0], p[1]))
    def detect(s, logic):
        for mid, T_bm in WORLD.items():
            T_cm = inv_T(s.cam) @ T_bm; x, y, z = T_cm[:3,3]
            if z < 0.1 or z > 1.5 or abs(x/z) > CX/FX or abs(y/z) > CY/FX: continue
            if T_cm[2,2] > -math.cos(math.radians(65)): continue      # marker non rivolto alla camera
            noise_p = rng.normal(0, 0.001 + 0.004*z*z, 3)
            ax = rng.normal(0, math.radians(0.5)*z, 3); th = np.linalg.norm(ax)
            K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])/max(th,1e-12)
            Rn = np.eye(3)+math.sin(th)*K+(1-math.cos(th))*K@K
            T_meas = make_T(T_cm[:3,:3] @ Rn, T_cm[:3,3]+noise_p)
            logic.on_detection(s.t, mid, s.cam @ T_meas, T_meas)

print("===== CASO 1: CLIK ideale =====")
io = FakeIO(); logic = InspectionLogic(dict(DEFAULTS), io)
dt = 0.05
while logic.state != "DONE" and io.t < 900:
    io.step(dt); io.t += dt; io.detect(logic); logic.tick()
print("\ntempo totale", round(io.t,1), "s; target inviati", io.sent, "rifiutati dal CLIK", io.rejected,
      "; raggio minimo camera", round(io.min_r,3))
for r in io.rows: print(r)

print("\n===== CASO 2: errore sistematico del CLIK, marker 202 rimosso =====")
# caso 2: CLIK con errore sistematico (7 mm lungo X camera, 1 grado) su OGNI comando; 202 sparisce dopo la mappatura
B = make_T(rot_x(math.radians(1.0)), [0.007, 0, 0])
class BiasIO(FakeIO):
    def send_target(s, T):
        FakeIO.send_target(s, T @ B)
io = BiasIO(); logic = InspectionLogic(dict(DEFAULTS), io)
dt = 0.05
while logic.state != "DONE" and io.t < 900:
    io.step(dt); io.t += dt
    if logic.state == "NEXT_MARKER": WORLD.pop(202, None)
    io.detect(logic); logic.tick()
for r in io.rows: print(r["id"], r["esito"], r["errore_pos_mm"], r["errore_angolo_deg"], "correzioni", r["correzioni"])
