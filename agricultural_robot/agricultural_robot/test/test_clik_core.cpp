// Test del CLIK sulla catena UR5e costruita dai parametri DH (senza ROS).
#include "agricultural_robot/clik_core.hpp"
#include <cstdio>
using namespace agricultural_robot;

int main()
{
  // UR5e da parametri DH ufficiali
  const double d[6] = {0.1625, 0, 0, 0.1333, 0.0997, 0.0996};
  const double a[6] = {0, -0.425, -0.3922, 0, 0, 0};
  const double al[6] = {M_PI / 2, 0, 0, M_PI / 2, -M_PI / 2, 0};
  KDL::Chain ch;
  for (int i = 0; i < 6; ++i)
    ch.addSegment(KDL::Segment(KDL::Joint(KDL::Joint::RotZ), KDL::Frame::DH(a[i], al[i], d[i], 0)));

  ClikParams p;
  ClikController c(ch, p);
  Eigen::VectorXd q0(6);
  q0 << 0.0, -1.57, 1.57, -1.57, -1.57, 0.0;
  c.reset(q0);
  KDL::Frame start = c.forward(q0);
  KDL::Frame target = start;
  target.p += KDL::Vector(0.10, -0.15, -0.05);
  target.M = KDL::Rotation::RotZ(0.4) * KDL::Rotation::RotX(0.3) * start.M;
  c.setTarget(target);

  const double dt = 0.005;
  double max_err = 0, max_qd = 0, max_qdd = 0, min_w = 1e9;
  for (int k = 0; k < 2000; ++k) {
    if (!c.step(dt)) {printf("step fallito\n"); return 1;}
    max_err = std::max(max_err, c.error().head<3>().norm());
    max_qd = std::max(max_qd, c.qd().cwiseAbs().maxCoeff());
    max_qdd = std::max(max_qdd, c.qdd().cwiseAbs().maxCoeff());
    min_w = std::min(min_w, c.manipulability());
    if (k % 400 == 0)
      printf("t=%.2f  |e_p|=%.2e m  |e_o|=%.2e  w=%.4f\n", k * dt,
        c.error().head<3>().norm(), c.error().tail<3>().norm(), c.manipulability());
  }
  KDL::Frame fin = c.forward(c.q());
  KDL::Vector ax;
  double ang = (target.M * fin.M.Inverse()).GetRotAngle(ax);
  printf("finale: errore pos %.2e m, errore ori %.2e rad\n", (target.p - fin.p).Norm(), ang);
  printf("max errore di inseguimento %.2e m, max |qd| %.3f, max |qdd| %.3f, min w %.4f\n",
    max_err, max_qd, max_qdd, min_w);
  return 0;
}
