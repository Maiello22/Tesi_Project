#!/usr/bin/env python3
"""
inspection_node: ispezione automatica di piu' marker ArUco (mese 3).

Fasi
  1. MAPPATURA   giro di scansione: la camera viene portata a scan_radius dall'asse della base e a
                 quota scan_height, con l'orientamento iniziale, e poi ruotata attorno all'asse Z
                 della base (angoli scan_angles_deg); a ogni posa si raccolgono le rilevazioni e si
                 salva la posizione di ogni marker in base_link.
  2. CLASSIFICA  marker visti almeno min_detections volte; raggiungibile se la posa di foto della
                 camera (marker + standoff, perpendicolare) dista dalla spalla <= max_reach.
  3. VISITA      dal piu' vicino al successivo piu' vicino; per ognuno avvicinamento a raffinamenti
                 successivi (approach_distances), con una nuova stima del marker a ogni tappa.
  4. VERIFICA    da fermo: il marker deve essere a standoff sull'asse ottico e perpendicolare;
                 altrimenti correzione (max_corrections). Poi FOTO.
                 Nota: vista quasi frontale, l'inclinazione stimata da un solo marker ha un errore
                 di 6-8 gradi (ambiguita' della posa planare, IPPE); da 30 gradi di obliquita'
                 scende sotto 1 grado. Per questo, prima della foto, la camera osserva il marker da
                 due VISTE OBLIQUE (oblique_angles_deg, ruotando attorno alla verticale passante per
                 il marker) e la normale si stima solo dalle osservazioni con angolo di vista
                 >= min_view_angle_deg. La vista frontale corregge solo la posizione. Se e' noto il
                 ground truth (gt_markers), il rapporto riporta anche l'errore vero.
  5. RAPPORTO    CSV con esito, errori finali e percorso della foto; ritorno alla prima posa di
                 scansione (la posa iniziale del robot puo' essere fuori dal limite max_reach).

Ingressi : /aruco/markers (aruco_msgs/MarkerArray, pose nel frame ottico), TF, /zed/left/image
Uscite   : /clik/target_pose (PoseStamped in base_link), /inspection/state (String), foto + CSV
"""

import csv
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

# ===================================================================================
# Geometria (numpy, senza ROS): trasformazioni omogenee 4x4 e quaternioni (x, y, z, w)
# ===================================================================================


def quat_to_mat(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return ((R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def make_T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_T(T):
    R, p = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ p)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_angle(R):
    """Angolo della rotazione R (rad)."""
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))


def mean_pose(Ts):
    """Media di pose vicine: posizione media, quaternioni sommati nello stesso emisfero."""
    p = np.mean([T[:3, 3] for T in Ts], axis=0)
    q0 = np.array(mat_to_quat(Ts[0][:3, :3]))
    acc = np.zeros(4)
    for T in Ts:
        q = np.array(mat_to_quat(T[:3, :3]))
        acc += q if np.dot(q, q0) >= 0 else -q
    return make_T(quat_to_mat(*(acc / np.linalg.norm(acc))), p)


# ===================================================================================
# Logica dell'ispezione (senza ROS: le funzioni di I/O sono iniettate)
# ===================================================================================

@dataclass
class MarkerRecord:
    marker_id: int
    obs: list = field(default_factory=list)       # pose T_base_M osservate in mappatura
    status: str = "da visitare"
    reach: float = float("nan")
    est: np.ndarray = None                         # stima corrente T_base_M
    pos_err_mm: float = float("nan")
    lat_err_mm: float = float("nan")
    dist_m: float = float("nan")
    ang_err_deg: float = float("nan")
    refinements: int = 0
    corrections: int = 0
    photo: str = ""
    pool: list = field(default_factory=list)      # (T_base_M, angolo di vista in gradi)
    oblique_done: bool = False
    oblique_obs: int = 0
    gt: dict = field(default_factory=dict)         # errori rispetto al ground truth


class InspectionLogic:
    """Macchina a stati dell'ispezione. tick() va chiamato periodicamente (es. 20 Hz)."""

    def __init__(self, p, io):
        self.p = p
        self.io = io            # now(), cam_pose(), send_target(T), save_photo(name), log(msg), state(msg)
        self.state = "INIT"
        self.t_state = 0.0
        self.markers = {}
        self.dets = []          # rilevazioni recenti: (t, id, T_base_M, T_cam_M)
        self.home = None
        self.target = None
        self.scan_targets = []
        self.scan_idx = 0
        self.queue = []
        self.current = None
        self.stages = []
        self.stage_idx = 0
        self.stable_since = None
        self.report_rows = []
        self.waypoints = []
        self.current_wp = None
        self.t_sent = 0.0
        self.last_cam = None
        self.wind = 0.0         # rotazione attorno alla base accumulata dall'inizio [rad]
        self.obl_targets = []
        self.obl_idx = 0

    # ---------------- ingressi ----------------
    def on_detection(self, t, marker_id, T_base_M, T_cam_M):
        self.dets.append((t, marker_id, T_base_M, T_cam_M))
        horizon = t - 5.0
        if self.dets and self.dets[0][0] < horizon:
            self.dets = [d for d in self.dets if d[0] >= horizon]

    # ---------------- supporto ----------------
    def _goto(self, state):
        self.state = state
        self.t_state = self.io.now()
        self.stable_since = None
        self.io.state(state if self.current is None else f"{state} (marker {self.current.marker_id})")

    def _send(self, T):
        """Invia il target al CLIK. Il CLIK interpola in linea retta: se il target e' molto
        ruotato attorno alla base rispetto alla camera, il segmento passerebbe vicino al robot.
        In quel caso si inseriscono pose intermedie (il target ruotato attorno a Z della base,
        a passi di al massimo via_step_deg) e solo dopo si va al target."""
        self.stable_since = None
        self.waypoints = []
        cam = self.io.cam_pose()
        if cam is not None:
            a0 = math.atan2(cam[1, 3], cam[0, 3])
            a1 = math.atan2(T[1, 3], T[0, 3])
            da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            # Avvolgimento: la base non deve accumulare giri nello stesso verso (il giunto della
            # spalla ha un limite di +-360 gradi). Si tiene conto della rotazione totale rispetto
            # all'inizio e, se il verso piu' breve la porterebbe oltre max_wind_deg, si gira
            # dall'altra parte.
            lim = math.radians(self.p["max_wind_deg"])
            if abs(self.wind + da) > lim:
                alt = da - math.copysign(2 * math.pi, da)
                if abs(self.wind + alt) < abs(self.wind + da):
                    da = alt
            self.wind += da
            # pose intermedie: il TARGET ruotato all'indietro verso la posizione di partenza
            # (stessa distanza e quota del target, quindi raggiungibili come il target)
            n = int(math.ceil(abs(da) / math.radians(self.p["via_step_deg"])))
            for k in range(1, n):
                self.waypoints.append(make_T(rot_z(-da * (n - k) / n), [0, 0, 0]) @ T)
        self.waypoints.append(T)
        self.target = T
        self._dispatch()

    def _dispatch(self):
        self.current_wp = self.waypoints.pop(0)
        self.t_sent = self.io.now()
        self.io.send_target(self.current_wp)

    def _motion_timed_out(self):
        if self.io.now() - self.t_sent <= self.p["motion_timeout"]:
            return False
        cam = self.io.cam_pose()
        if cam is not None and self.target is not None:
            dp = 1000 * np.linalg.norm(cam[:3, 3] - self.target[:3, 3])
            da = math.degrees(rot_angle(cam[:3, :3].T @ self.target[:3, :3]))
            self.io.log(f"Timeout: target {np.round(self.target[:3, 3], 3)} (asse ottico "
                        f"{np.round(self.target[:3, 2], 2)}), camera {np.round(cam[:3, 3], 3)}; "
                        f"scarto {dp:.0f} mm, {da:.1f} deg; tolleranze {1000 * self.p['arrive_pos_tol']:.0f} mm, "
                        f"{self.p['arrive_ang_tol_deg']:.0f} deg")
        return True

    def _arrived(self):
        """La camera reale (TF) ha raggiunto il target ed e' ferma da arrive_stable secondi."""
        cam = self.io.cam_pose()
        if cam is None or self.target is None:
            return False
        if self.waypoints:     # posa intermedia: basta passarci vicino, poi la successiva
            if np.linalg.norm(cam[:3, 3] - self.current_wp[:3, 3]) < self.p["via_tol"]:
                self._dispatch()
            return False
        # arrivato = vicino al target (tolleranza larga) E fermo (velocita' della camera ~ 0).
        # L'errore fine non si giudica qui ma nella verifica visiva sul marker.
        dp = np.linalg.norm(cam[:3, 3] - self.target[:3, 3])
        da = math.degrees(rot_angle(cam[:3, :3].T @ self.target[:3, :3]))
        now = self.io.now()
        moving = True
        if self.last_cam is not None and now > self.last_cam[0]:
            v = np.linalg.norm(cam[:3, 3] - self.last_cam[1][:3, 3]) / (now - self.last_cam[0])
            moving = v > self.p["still_vel"]
        self.last_cam = (now, cam)
        started = now - self.t_sent >= self.p["min_motion_time"]   # il quintico dura almeno 1 s
        if started and dp < self.p["arrive_pos_tol"] and da < self.p["arrive_ang_tol_deg"] and not moving:
            if self.stable_since is None:
                self.stable_since = now
            return now - self.stable_since >= self.p["arrive_stable"]
        self.stable_since = None
        return False

    def _timed_out(self, limit):
        return self.io.now() - self.t_state > limit

    def _recent(self, marker_id, since):
        return [d for d in self.dets if d[1] == marker_id and d[0] >= since]

    def _photo_pose(self, T_base_M, d):
        """Posa desiderata della camera: d metri davanti al marker, perpendicolare."""
        R_M_C = quat_to_mat(*self.p["standoff_quat"])
        return T_base_M @ make_T(R_M_C, [0.0, 0.0, d])

    def _reach(self, T):
        shoulder = np.array([0.0, 0.0, self.p["shoulder_height"]])
        return float(np.linalg.norm(T[:3, 3] - shoulder))

    # ---------------- macchina a stati ----------------
    def tick(self):
        s = self.state
        if s == "INIT":
            cam = self.io.cam_pose()
            if cam is None or not self.io.ready() or self.io.now() - self.t_state < self.p["start_delay"]:
                return
            self.home = cam
            self._build_scan()
            self.io.log(f"Posa iniziale della camera: {np.round(cam[:3, 3], 3)} "
                        f"({self._reach(cam):.3f} m dalla spalla); {len(self.scan_targets)} pose di scansione")
            if not self.scan_targets:
                self.io.log("Nessuna posa di scansione raggiungibile: ispezione annullata")
                self._write_report()
                self._goto("DONE")
                return
            self.home = self.scan_targets[0]      # posa di ritorno a fine ispezione
            self.scan_idx = 0
            self._send(self.scan_targets[0])
            self._goto("SCAN_MOVE")

        elif s == "SCAN_MOVE":
            if self._arrived():
                self._goto("SCAN_DWELL")
            elif self._motion_timed_out():
                self.io.log(f"Posa di scansione {self.scan_idx} non raggiunta, la salto")
                self._next_scan()

        elif s == "SCAN_DWELL":
            if self._timed_out(self.p["scan_dwell"]):
                for (t, mid, T_bm, T_cm) in self._recent_any(self.t_state):
                    rec = self.markers.setdefault(mid, MarkerRecord(mid))
                    rec.obs.append(T_bm)
                    rec.pool.append((T_bm, self._view_angle(T_cm)))
                self._next_scan()

        elif s == "PLAN":
            self._plan()

        elif s == "NEXT_MARKER":
            self._next_marker()

        elif s == "STAGE_MOVE":
            if self._arrived():
                self._goto("STAGE_MEASURE")
            elif self._motion_timed_out():
                self._finish_marker("posa non raggiunta")

        elif s == "STAGE_MEASURE":
            self._stage_measure()

        elif s == "OBL_MOVE":
            if self._arrived():
                self._goto("OBL_MEASURE")
            elif self._motion_timed_out():
                self.io.log(f"Vista obliqua {self.obl_idx} non raggiunta, la salto")
                self._next_oblique()

        elif s == "OBL_MEASURE":
            self._oblique_measure()

        elif s == "HOME":
            if self._arrived() or self._motion_timed_out():
                self._write_report()
                self._goto("DONE")

    def _recent_any(self, since):
        return [d for d in self.dets if d[0] >= since]

    # ---------------- mappatura ----------------
    def _build_scan(self):
        """Pose di scansione: la camera mantiene l'orientamento iniziale, inclinato di scan_tilt_deg
        verso il basso, e viene portata a scan_radius dall'asse della base (nella direzione in cui
        guarda) e a quota scan_height; poi la posa viene ruotata attorno all'asse Z della base.
        Le pose oltre max_reach dalla spalla vengono scartate (il CLIK le rifiuterebbe)."""
        T0 = self.home.copy()
        # inclinazione: rotazione attorno a X del frame ottico; -tilt porta Z (vista) verso +Y (basso)
        T0[:3, :3] = T0[:3, :3] @ rot_x(-math.radians(self.p["scan_tilt_deg"]))
        p = T0[:3, 3].copy()
        if self.p["scan_radius"] > 0.0:
            # direzione orizzontale dell'asse ottico (la camera guarda verso l'esterno)
            v = T0[:2, 2].copy()
            if np.linalg.norm(v) < 1e-3:
                v = p[:2].copy()
            v /= np.linalg.norm(v)
            p[0], p[1] = self.p["scan_radius"] * v[0], self.p["scan_radius"] * v[1]
        if self.p["scan_height"] > 0.0:
            p[2] = self.p["scan_height"]
        T0[:3, 3] = p
        self.scan_targets = []
        for a in self.p["scan_angles_deg"]:
            T = make_T(rot_z(math.radians(a)), [0, 0, 0]) @ T0
            if self._reach(T) > self.p["max_reach"]:
                self.io.log(f"Posa di scansione a {a:.0f} deg scartata: {self._reach(T):.3f} m "
                            f"dalla spalla (> {self.p['max_reach']:.2f}); riduci scan_radius/scan_height")
                continue
            self.scan_targets.append(T)

    def _next_scan(self):
        self.scan_idx += 1
        if self.scan_idx < len(self.scan_targets):
            self._send(self.scan_targets[self.scan_idx])
            self._goto("SCAN_MOVE")
        else:
            self._goto("PLAN")

    # ---------------- classificazione e ordine di visita ----------------
    def _plan(self):
        d_final = self.p["approach_distances"][-1]
        reachable = []
        for m in self.markers.values():
            if len(m.obs) < self.p["min_detections"]:
                m.status = "scartato (poche rilevazioni)"
                continue
            m.est = make_T(self._orientation(m), mean_pose(m.obs)[:3, 3])
            m.reach = self._reach(self._photo_pose(m.est, d_final))
            if m.reach > self.p["max_reach"]:
                m.status = "fuori portata"
            else:
                reachable.append(m)
        self.io.log("Mappatura conclusa: " + (", ".join(
            f"id {m.marker_id} ({len(m.obs)} oss., {m.status}, spalla {m.reach:.2f} m)"
            for m in sorted(self.markers.values(), key=lambda m: m.marker_id)) or "nessun marker"))
        # ordine: vicino piu' vicino, partendo dalla posizione attuale della camera
        cam = self.io.cam_pose()
        pos = cam[:3, 3] if cam is not None else self.home[:3, 3]
        self.queue = []
        while reachable:
            nxt = min(reachable, key=lambda m: np.linalg.norm(
                self._photo_pose(m.est, d_final)[:3, 3] - pos))
            reachable.remove(nxt)
            self.queue.append(nxt)
            pos = self._photo_pose(nxt.est, d_final)[:3, 3]
        self.io.log("Ordine di visita: " + (", ".join(str(m.marker_id) for m in self.queue) or "-"))
        self._goto("NEXT_MARKER")

    # ---------------- visita di un marker ----------------
    def _next_marker(self):
        if not self.queue:
            self.current = None
            self._send(self.home)
            self._goto("HOME")
            return
        self.current = self.queue.pop(0)
        cam = self.io.cam_pose()
        dist_now = np.linalg.norm(self.current.est[:3, 3] - cam[:3, 3])
        # tappe: solo quelle piu' vicine della distanza attuale; l'ultima (standoff) sempre
        self.stages = [d for d in self.p["approach_distances"][:-1] if d < dist_now - 0.05]
        self.stages.append(self.p["approach_distances"][-1])
        self.stage_idx = 0
        self.io.log(f"Marker {self.current.marker_id}: avvicinamento in {len(self.stages)} tappe "
                    f"{self.stages} (distanza attuale {dist_now:.2f} m)")
        self._send_stage()

    def _stage_measure(self):
        m = self.current
        settle = self.p["settle_time"]
        if self.io.now() - self.t_state < settle + self.p["measure_window"]:
            return
        dets = self._recent(m.marker_id, self.t_state + settle)
        if len(dets) < self.p["min_frames"]:
            if self._timed_out(settle + 3 * self.p["measure_window"]):
                final = self.stage_idx == len(self.stages) - 1
                if final:
                    self._finish_marker("perso")
                else:                       # tappa intermedia: si prosegue con la stima precedente
                    self.io.log(f"Marker {m.marker_id} non visto alla tappa {self.stage_idx}, "
                                "proseguo con la stima precedente")
                    self._advance_stage()
            return

        # stima rinnovata in base_link: posizione dalla tappa attuale (la piu' vicina, quindi la
        # piu' precisa), orientamento dalla media di tutte le osservazioni accumulate
        m.pool.extend((d[2], self._view_angle(d[3])) for d in dets)
        p_now = mean_pose([d[2] for d in dets])[:3, 3]
        m.est = make_T(self._orientation(m), p_now)
        m.refinements += 1
        final = self.stage_idx == len(self.stages) - 1
        if not final:
            self._advance_stage()
            return

        # ---- verifica alla tappa finale: errore nel frame camera ----
        T_cm = mean_pose([d[3] for d in dets])
        d = self.stages[-1]
        err = T_cm[:3, 3] - np.array([0.0, 0.0, d])
        ang = math.degrees(math.acos(max(-1.0, min(1.0, -T_cm[2, 2]))))   # Z marker vs -Z camera
        m.pos_err_mm = 1000 * float(np.linalg.norm(err))
        m.lat_err_mm = 1000 * float(math.hypot(err[0], err[1]))
        m.dist_m = float(T_cm[2, 3])
        m.ang_err_deg = ang
        # l'angolo misurato in vista frontale non e' attendibile (6-8 gradi di errore): se la normale
        # viene dalle viste oblique, la verifica dell'angolo e' gia' fatta e qui conta la posizione
        oblique_ok = m.oblique_obs >= self.p["min_frames"]
        ang_ok = ang <= self.p["verify_ang_tol_deg"] or oblique_ok
        ok = m.pos_err_mm <= 1000 * self.p["verify_pos_tol"] and ang_ok
        self.io.log(f"Marker {m.marker_id}: verifica errore {m.pos_err_mm:.1f} mm, {ang:.2f} deg"
                    f"{' (angolo frontale indicativo, normale dalle viste oblique)' if oblique_ok else ''}"
                    f" -> {'OK' if ok else 'da correggere'}")
        if ok:
            m.photo = self.io.save_photo(f"marker_{m.marker_id}")
            self._ground_truth(m)
            self._finish_marker("fotografato")
        elif m.corrections < self.p["max_corrections"]:
            # correzione: la posa voluta (dalla stima appena rinnovata) viene compensata con lo
            # scarto tra comando precedente e posa reale, cosi' si recupera anche un errore
            # sistematico del CLIK:  T_cmd' = T_des * T_reale^-1 * T_cmd
            m.corrections += 1
            T_des = self._photo_pose(m.est, d)
            cam = self.io.cam_pose()
            self._send(T_des @ inv_T(cam) @ self.target if cam is not None else T_des)
            self._goto("STAGE_MOVE")
        else:
            m.photo = self.io.save_photo(f"marker_{m.marker_id}_fuori_tolleranza")
            self._ground_truth(m)
            self._finish_marker("fotografato (fuori tolleranza)")

    def _gt_marker(self, marker_id):
        """Centro c e normale n veri del marker (da gt_markers), oppure None."""
        gt = self.p["gt_markers"]
        for i in range(0, len(gt) - 3, 4):
            if int(gt[i]) == marker_id:
                R = rot_z(gt[i + 3])
                c = np.array([gt[i + 1], gt[i + 2], 0.0]) + R @ np.array(self.p["gt_marker_offset"])
                return c, R @ np.array([-1.0, 0.0, 0.0])
        return None

    def _ground_truth(self, m):
        """Errore vero della posa di foto, se la posa del marker e' nota (gt_markers: id, x, y, yaw
        del modello nel mondo, come nel file .sdf; base_link coincide con l'origine del mondo)."""
        gt = self.p["gt_markers"]
        if len(gt) < 4:
            return
        for i in range(0, len(gt) - 3, 4):
            if int(gt[i]) != m.marker_id:
                continue
            x, y, yaw = gt[i + 1], gt[i + 2], gt[i + 3]
            R = rot_z(yaw)
            c = np.array([x, y, 0.0]) + R @ np.array(self.p["gt_marker_offset"])   # centro marker
            n = R @ np.array([-1.0, 0.0, 0.0])          # normale uscente dalla faccia (-X modello)
            cam = self.io.cam_pose()
            if cam is None:
                return
            d = self.p["approach_distances"][-1]
            v = cam[:3, 3] - c
            along = float(np.dot(v, n))
            lat = v - along * n
            m.gt.update({
                "gt_errore_pos_mm": round(1000 * float(np.linalg.norm(cam[:3, 3] - (c + d * n))), 2),
                "gt_distanza_m": round(along, 4),
                "gt_laterale_mm": round(1000 * float(np.linalg.norm(lat)), 2),
                "gt_angolo_deg": round(math.degrees(math.acos(
                    max(-1.0, min(1.0, float(np.dot(cam[:3, 2], -n)))))), 3)})
            self.io.log(f"Marker {m.marker_id} ground truth: errore {m.gt['gt_errore_pos_mm']:.1f} mm "
                        f"(laterale {m.gt['gt_laterale_mm']:.1f} mm, distanza {m.gt['gt_distanza_m']:.3f} m), "
                        f"angolo {m.gt['gt_angolo_deg']:.2f} deg")
            return

    def _advance_stage(self):
        self.stage_idx += 1
        self._send_stage()

    def _send_stage(self):
        """Invia la tappa corrente; prima della tappa finale esegue le viste oblique."""
        m = self.current
        final = self.stage_idx == len(self.stages) - 1
        if final and not m.oblique_done and len(self.p["oblique_angles_deg"]) > 0:
            m.oblique_done = True
            self.obl_targets = []
            for a in self.p["oblique_angles_deg"]:
                T = self._oblique_pose(m.est, math.radians(a), self.p["oblique_distance"])
                if self._reach(T) <= self.p["max_reach"]:
                    self.obl_targets.append((a, T))
                else:
                    self.io.log(f"Marker {m.marker_id}: vista obliqua a {a:+.0f} deg fuori portata "
                                f"({self._reach(T):.2f} m), la salto")
            if self.obl_targets:
                self.obl_idx = 0
                self._send(self.obl_targets[0][1])
                self._goto("OBL_MOVE")
                return
        self._send(self._photo_pose(m.est, self.stages[self.stage_idx]))
        self._goto("STAGE_MOVE")

    def _oblique_pose(self, T_base_M, angle, d):
        """Posa a distanza d dal marker, ruotata di angle attorno alla verticale (Z della base)
        passante per il centro del marker: la camera continua a guardare il centro, ma con un
        angolo di vista pari ad angle."""
        c = T_base_M[:3, 3]
        Rz = make_T(rot_z(angle), [0, 0, 0])
        return make_T(np.eye(3), c) @ Rz @ make_T(np.eye(3), -c) @ self._photo_pose(T_base_M, d)

    def _oblique_measure(self):
        m = self.current
        settle = self.p["settle_time"]
        if self.io.now() - self.t_state < settle + self.p["measure_window"]:
            return
        dets = self._recent(m.marker_id, self.t_state + settle)
        if len(dets) < self.p["min_frames"] and not self._timed_out(settle + 3 * self.p["measure_window"]):
            return
        if dets:
            views = [self._view_angle(d[3]) for d in dets]
            m.pool.extend((d[2], v) for d, v in zip(dets, views))
            m.oblique_obs += len(dets)
            self.io.log(f"Marker {m.marker_id}: vista obliqua {self.obl_targets[self.obl_idx][0]:+.0f} deg, "
                        f"{len(dets)} frame, angolo di vista misurato {np.mean(views):.1f} deg")
        else:
            self.io.log(f"Marker {m.marker_id}: nessuna rilevazione nella vista obliqua {self.obl_idx}")
        self._next_oblique()

    def _next_oblique(self):
        self.obl_idx += 1
        if self.obl_idx < len(self.obl_targets):
            self._send(self.obl_targets[self.obl_idx][1])
            self._goto("OBL_MOVE")
            return
        m = self.current
        old = m.est[:3, 2].copy()
        m.est = make_T(self._orientation(m), m.est[:3, 3])
        dn = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(old, m.est[:3, 2]))))))
        self.io.log(f"Marker {m.marker_id}: normale aggiornata dalle viste oblique ({dn:.2f} deg di correzione)")
        self._send_stage()

    def _view_angle(self, T_cm):
        """Angolo tra asse ottico e normale del marker (0 = vista frontale) [deg]."""
        return math.degrees(math.acos(max(-1.0, min(1.0, -float(T_cm[2, 2])))))

    def _orientation(self, m):
        """Orientamento del marker: media delle sole osservazioni oblique (angolo di vista
        >= min_view_angle_deg); se sono meno di min_frames, media di tutte."""
        sel = [T for T, a in m.pool if a >= self.p["min_view_angle_deg"]]
        if len(sel) < self.p["min_frames"]:
            sel = [T for T, _ in m.pool] or [m.est]
        return mean_pose(sel)[:3, :3]

    def _finish_marker(self, status):
        self.current.status = status
        self.io.log(f"Marker {self.current.marker_id}: {status}")
        self._goto("NEXT_MARKER")

    # ---------------- rapporto ----------------
    def _write_report(self):
        rows = []
        for m in sorted(self.markers.values(), key=lambda m: m.marker_id):
            if m.est is not None:
                px, py, pz = (round(float(v), 4) for v in m.est[:3, 3])
                nx, ny, nz = (round(float(v), 4) for v in m.est[:3, 2])
                g = self._gt_marker(m.marker_id)
                if g is not None:     # errore della stima del marker, per tutti (anche fuori portata)
                    m.gt["gt_errore_marker_mm"] = round(1000 * float(np.linalg.norm(m.est[:3, 3] - g[0])), 2)
                    m.gt["gt_errore_normale_deg"] = round(math.degrees(math.acos(
                        max(-1.0, min(1.0, float(np.dot(m.est[:3, 2], g[1])))))), 3)
            else:
                px = py = pz = nx = ny = nz = float("nan")
            rows.append({
                "id": m.marker_id, "esito": m.status,
                "x_m": px, "y_m": py, "z_m": pz,           # centro del marker stimato, base_link
                "nx": nx, "ny": ny, "nz": nz,             # normale stimata (uscente dalla faccia)
                "osservazioni": len(m.obs),
                "distanza_spalla_m": round(m.reach, 3),
                "distanza_finale_m": round(m.dist_m, 4),
                "errore_pos_mm": round(m.pos_err_mm, 2),
                "errore_laterale_mm": round(m.lat_err_mm, 2),
                "errore_angolo_deg": round(m.ang_err_deg, 3),
                "raffinamenti": m.refinements, "correzioni": m.corrections,
                "frame_obliqui": m.oblique_obs,
                **{k: m.gt.get(k, float("nan")) for k in
                   ("gt_errore_marker_mm", "gt_errore_normale_deg", "gt_errore_pos_mm", "gt_distanza_m",
                    "gt_laterale_mm", "gt_angolo_deg")},
                "foto": m.photo})
        self.report_rows = rows
        self.io.write_report(rows)
        self.io.log("RAPPORTO: " + "; ".join(f"id {r['id']} {r['esito']}" for r in rows) if rows
                    else "RAPPORTO: nessun marker trovato")


# ===================================================================================
# Nodo ROS 2
# ===================================================================================

DEFAULTS = {
    "base_frame": "base_link",
    "camera_frame": "zed_left_camera_frame_optical",
    "markers_topic": "/aruco/markers",
    "image_topic": "/zed/left/image",
    "target_topic": "/clik/target_pose",
    "standoff_quat": [0.7071068, 0.7071068, 0.0, 0.0],   # orientamento camera nel frame marker
    "approach_distances": [0.55, 0.30],                    # tappe; l'ultima e' la distanza della foto
    "oblique_angles_deg": [-30.0, 30.0],                   # viste oblique prima della foto ([] = nessuna)
    "oblique_distance": 0.45,
    "min_view_angle_deg": 15.0,                            # sotto: l'osservazione non stima la normale
    "scan_angles_deg": [-120.0, -60.0, 0.0, 60.0, 120.0, 180.0],   # in ordine: rotazione monotona
    "scan_radius": 0.45,                                   # [m] 0 = raggio della posa iniziale
    "scan_height": 0.75,                                   # [m] 0 = quota della posa iniziale
    "scan_tilt_deg": 5.0,
    "scan_dwell": 1.5,
    "min_detections": 3,
    "shoulder_height": 0.1625,
    "max_reach": 0.95,
    "arrive_pos_tol": 0.02,
    "arrive_ang_tol_deg": 5.0,
    "still_vel": 0.002,
    "min_motion_time": 1.2,
    "arrive_stable": 0.4,
    "motion_timeout": 30.0,
    "via_step_deg": 60.0,
    "max_wind_deg": 190.0,
    "via_tol": 0.03,
    "start_delay": 3.0,
    "settle_time": 0.3,
    "measure_window": 1.0,
    "min_frames": 5,
    "verify_pos_tol": 0.005,
    "verify_ang_tol_deg": 8.0,
    "gt_markers": [-1.0],                                  # [id, x, y, yaw, id, x, y, yaw, ...]
    "gt_marker_offset": [-0.016, 0.0, 0.75],               # centro del marker nel frame del modello
    "max_corrections": 2,
    "photo_dir": "~/ros2_ws/src/inspection_results",      # cartella montata dall'host (~/tesi_ws)
    "rate": 20.0,
}


def main(args=None):
    import rclpy
    import cv2
    from aruco_msgs.msg import MarkerArray
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener

    def tf_to_T(tf):
        t, q = tf.transform.translation, tf.transform.rotation
        return make_T(quat_to_mat(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])

    class InspectionNode(Node):
        def __init__(self):
            super().__init__("inspection_node")
            self.p = {k: self.declare_parameter(k, v).value for k, v in DEFAULTS.items()}
            self.p["photo_dir"] = os.path.expanduser(self.p["photo_dir"])
            run = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(self.p["photo_dir"], run)
            os.makedirs(self.run_dir, exist_ok=True)
            # la cartella sta dentro src/: COLCON_IGNORE dice a colcon di non esplorarla
            ignore = os.path.join(self.p["photo_dir"], "COLCON_IGNORE")
            if not os.path.exists(ignore):
                open(ignore, "w").close()

            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.bridge = CvBridge()
            self.last_image = None
            self.target_pub = self.create_publisher(PoseStamped, self.p["target_topic"], 10)
            self.state_pub = self.create_publisher(String, "/inspection/state", 10)
            self.create_subscription(MarkerArray, self.p["markers_topic"], self.on_markers, 10)
            self.create_subscription(Image, self.p["image_topic"], self.on_image,
                                     qos_profile_sensor_data)

            self.logic = InspectionLogic(self.p, self)
            self.create_timer(1.0 / self.p["rate"], self.logic.tick)
            self.get_logger().info(f"inspection_node avviato, foto e rapporto in {self.run_dir}")

        # ---- interfaccia usata da InspectionLogic ----
        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def ready(self):
            """Il clik_node e' in ascolto su /clik/target_pose."""
            return self.target_pub.get_subscription_count() > 0

        def cam_pose(self):
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.p["base_frame"], self.p["camera_frame"], Time())
                return tf_to_T(tf)
            except Exception:
                return None

        def send_target(self, T):
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.p["base_frame"]
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, T[:3, 3])
            q = mat_to_quat(T[:3, :3])
            (msg.pose.orientation.x, msg.pose.orientation.y,
             msg.pose.orientation.z, msg.pose.orientation.w) = map(float, q)
            self.target_pub.publish(msg)

        def save_photo(self, name):
            if self.last_image is None:
                return ""
            path = os.path.join(self.run_dir, name + ".png")
            cv2.imwrite(path, self.last_image)
            return path

        def write_report(self, rows):
            path = os.path.join(self.run_dir, "rapporto.csv")
            if rows:
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                self.get_logger().info(f"Rapporto salvato in {path}")

        def log(self, msg):
            self.get_logger().info(msg)

        def state(self, msg):
            self.get_logger().info(f"[stato] {msg}")
            self.state_pub.publish(String(data=msg))

        # ---- callback ----
        def on_image(self, msg):
            self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        def on_markers(self, msg):
            if not msg.markers:
                return
            frame = msg.header.frame_id or self.p["camera_frame"]
            try:     # posa della camera all'istante dello scatto
                T_bc = tf_to_T(self.tf_buffer.lookup_transform(
                    self.p["base_frame"], frame, Time.from_msg(msg.header.stamp)))
            except Exception:
                # TF non ancora disponibile per quell'istante: si usa l'ultima. Le rilevazioni
                # vengono usate solo a braccio fermo, quindi la differenza e' trascurabile.
                try:
                    T_bc = tf_to_T(self.tf_buffer.lookup_transform(
                        self.p["base_frame"], frame, Time()))
                except Exception:
                    return
            t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
            for m in msg.markers:
                q, pp = m.pose.pose.orientation, m.pose.pose.position
                T_cm = make_T(quat_to_mat(q.x, q.y, q.z, q.w), [pp.x, pp.y, pp.z])
                self.logic.on_detection(t, int(m.id), T_bc @ T_cm, T_cm)

    rclpy.init(args=args)
    node = InspectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()