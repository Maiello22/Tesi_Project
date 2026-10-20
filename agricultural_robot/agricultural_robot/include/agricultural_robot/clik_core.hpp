// CLIK del secondo ordine (Closed-Loop Inverse Kinematics) per una catena KDL.
//
// Legge di controllo (Siciliano et al., "Robotics: Modelling, Planning and Control", cap. 3.7):
//
//   qdd = J#(q) * ( xdd_d + K_D (xd_d - J q_dot) + K_P e - Jdot(q, q_dot) q_dot )
//
//   e   = [ p_d - p ;  e_o ]       e_o = parte vettoriale di Q_d * Q^-1 (errore in quaternione)
//   J#  = pseudo-inversa smorzata (Damped Least Squares) con smorzamento variabile
//         in funzione della manipolabilita' w = sqrt(det(J J^T))
//
// Accorgimenti per obiettivi fuori portata / vicino alle singolarita':
//   - l'errore cartesiano usato dalla legge e' saturato (max_pos_error, max_ori_error),
//     cosi' un obiettivo irraggiungibile non accumula una "spinta" enorme;
//   - termine di smorzamento nello spazio dei giunti  - (I - J# J) K_null q_dot :
//     smorza le direzioni che la DLS non controlla (quelle singolari), che altrimenti
//     restano senza attrito e fanno oscillare il braccio quando e' disteso.
//
// Il riferimento (p_d, Q_d) e le sue derivate sono generati da un polinomio quintico
// (velocita' e accelerazione nulle agli estremi) e da uno SLERP per l'orientamento.
// q e q_dot sono integrati internamente (CLIK "cinematico"): il robot riceve q o q_dot.
//
// Il file non dipende da ROS, cosi' puo' essere testato da solo.

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacdotsolver.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/frames.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/jntarrayvel.hpp>

namespace agricultural_robot
{

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

struct ClikParams
{
  double kp_pos = 100.0;      // [1/s^2]
  double kp_ori = 100.0;      // [1/s^2]
  double kd_pos = 20.0;       // [1/s]   (2*sqrt(kp) -> smorzamento critico)
  double kd_ori = 20.0;       // [1/s]
  double lambda_max = 0.10;   // smorzamento massimo della DLS
  double w_threshold = 0.03;  // soglia di manipolabilita' sotto cui si attiva lo smorzamento
  double max_joint_vel = 1.0; // [rad/s]
  double max_joint_acc = 4.0; // [rad/s^2]
  double max_lin_vel = 0.15;  // [m/s]   usata per calcolare la durata del moto
  double max_ang_vel = 0.6;   // [rad/s] usata per calcolare la durata del moto
  double min_duration = 1.0;  // [s]
  double max_pos_error = 0.05;  // [m]   saturazione dell'errore di posizione nella legge
  double max_ori_error = 0.30;  // [rad] (circa) saturazione dell'errore di orientamento
  double kd_null = 10.0;        // [1/s] smorzamento nelle direzioni singolari
};

class ClikController
{
public:
  ClikController(const KDL::Chain & chain, const ClikParams & params)
  : chain_(chain), params_(params), n_(chain.getNrOfJoints()),
    fk_(chain_), jac_solver_(chain_), jacdot_solver_(chain_),
    q_(n_), qd_(n_), qdd_(n_), jac_(n_)
  {
    jacdot_solver_.setRepresentation(KDL::ChainJntToJacDotSolver::HYBRID);
  }

  unsigned int nrJoints() const {return n_;}
  ClikParams & params() {return params_;}

  // Inizializza lo stato dal robot reale e mantiene la posa attuale.
  void reset(const Eigen::VectorXd & q_meas)
  {
    for (unsigned int i = 0; i < n_; ++i) {
      q_(i) = q_meas(i);
      qd_(i) = 0.0;
      qdd_(i) = 0.0;
    }
    fk_.JntToCart(q_, x_);
    start_ = target_ = x_;
    duration_ = 0.0;
    t_ = 0.0;
    initialized_ = true;
  }

  bool initialized() const {return initialized_;}

  // Nuovo obiettivo cartesiano (nel frame base della catena).
  // Il moto parte dalla posa desiderata attuale, cosi' un cambio di obiettivo a meta' moto
  // non provoca salti nel riferimento.
  void setTarget(const KDL::Frame & target)
  {
    KDL::Twist vel_unused, acc_unused;
    KDL::Frame current_ref;
    reference(current_ref, vel_unused, acc_unused);
    start_ = current_ref;
    target_ = target;

    const double dist = (target_.p - start_.p).Norm();
    const double angle = angleBetween(start_.M, target_.M);
    // Per il quintico la velocita' di picco e' 1.875 * distanza / T
    duration_ = std::max({params_.min_duration,
        1.875 * dist / params_.max_lin_vel,
        1.875 * angle / params_.max_ang_vel});
    t_ = 0.0;
  }

  // Un passo del CLIK. Restituisce false se la DLS non e' calcolabile.
  bool step(double dt)
  {
    if (!initialized_ || dt <= 0.0) {return false;}
    t_ += dt;

    // ---------------- Riferimento ----------------
    KDL::Frame x_d;
    KDL::Twist xd_d, xdd_d;
    reference(x_d, xd_d, xdd_d);
    desired_ = x_d;

    // ---------------- Cinematica ----------------
    fk_.JntToCart(q_, x_);
    jac_solver_.JntToJac(q_, jac_);
    KDL::JntArrayVel q_qd(q_, qd_);
    KDL::Twist jdot_qd;
    jacdot_solver_.JntToJacDot(q_qd, jdot_qd);

    const Eigen::MatrixXd J = jac_.data;
    Eigen::VectorXd qd_vec(n_);
    for (unsigned int i = 0; i < n_; ++i) {qd_vec(i) = qd_(i);}
    const Vector6d xd = J * qd_vec;  // velocita' attuale [v; w]

    // ---------------- Errori ----------------
    Vector6d e;
    e.head<3>() = toEigen(x_d.p - x_.p);
    e.tail<3>() = orientationError(x_d.M, x_.M);
    error_ = e;

    // Errore saturato usato nella legge (error_ resta quello vero, per la diagnostica)
    Vector6d e_sat = e;
    const double ep = e.head<3>().norm(), eo = e.tail<3>().norm();
    if (ep > params_.max_pos_error) {e_sat.head<3>() *= params_.max_pos_error / ep;}
    if (eo > params_.max_ori_error) {e_sat.tail<3>() *= params_.max_ori_error / eo;}

    Vector6d de = twistToEigen(xd_d) - xd;

    Matrix6d Kp = Matrix6d::Zero(), Kd = Matrix6d::Zero();
    Kp.diagonal() << params_.kp_pos, params_.kp_pos, params_.kp_pos,
      params_.kp_ori, params_.kp_ori, params_.kp_ori;
    Kd.diagonal() << params_.kd_pos, params_.kd_pos, params_.kd_pos,
      params_.kd_ori, params_.kd_ori, params_.kd_ori;

    const Vector6d u = twistToEigen(xdd_d) + Kd * de + Kp * e_sat - twistToEigen(jdot_qd);

    // ---------------- Pseudo-inversa smorzata ----------------
    const Matrix6d JJt = J * J.transpose();
    manipulability_ = std::sqrt(std::max(0.0, JJt.determinant()));
    double lambda2 = 0.0;
    if (manipulability_ < params_.w_threshold) {
      const double r = manipulability_ / params_.w_threshold;
      lambda2 = (1.0 - r * r) * params_.lambda_max * params_.lambda_max;
    }
    const Matrix6d A = JJt + lambda2 * Matrix6d::Identity();
    const Eigen::MatrixXd J_pinv = J.transpose() * A.ldlt().solve(Matrix6d::Identity());
    Eigen::VectorXd qdd_vec = J_pinv * u;

    // Smorzamento nelle direzioni non controllate dalla DLS
    const Eigen::MatrixXd N =
      Eigen::MatrixXd::Identity(n_, n_) - J_pinv * J;
    qdd_vec -= N * (params_.kd_null * qd_vec);
    if (!qdd_vec.allFinite()) {return false;}

    // ---------------- Saturazioni e integrazione ----------------
    for (unsigned int i = 0; i < n_; ++i) {
      double a = std::clamp(qdd_vec(i), -params_.max_joint_acc, params_.max_joint_acc);
      double v = std::clamp(qd_(i) + a * dt, -params_.max_joint_vel, params_.max_joint_vel);
      a = (v - qd_(i)) / dt;           // accelerazione effettivamente applicata
      q_(i) += qd_(i) * dt + 0.5 * a * dt * dt;
      qd_(i) = v;
      qdd_(i) = a;
    }
    return true;
  }

  // Comandi e diagnostica
  Eigen::VectorXd q() const {return q_.data;}
  Eigen::VectorXd qd() const {return qd_.data;}
  Eigen::VectorXd qdd() const {return qdd_.data;}
  const Vector6d & error() const {return error_;}
  const KDL::Frame & currentPose() const {return x_;}
  const KDL::Frame & desiredPose() const {return desired_;}
  const KDL::Frame & targetPose() const {return target_;}
  double manipulability() const {return manipulability_;}
  bool motionFinished() const {return t_ >= duration_;}
  double duration() const {return duration_;}

  KDL::Frame forward(const Eigen::VectorXd & q)
  {
    KDL::JntArray qa(n_);
    qa.data = q;
    KDL::Frame f;
    fk_.JntToCart(qa, f);
    return f;
  }

private:
  // Riferimento: quintico in s(t) su posizione e SLERP sull'orientamento
  void reference(KDL::Frame & x, KDL::Twist & v, KDL::Twist & a) const
  {
    double s = 1.0, sd = 0.0, sdd = 0.0;
    if (duration_ > 0.0 && t_ < duration_) {
      const double T = duration_, tau = t_ / T;
      const double tau2 = tau * tau, tau3 = tau2 * tau, tau4 = tau3 * tau, tau5 = tau4 * tau;
      s = 10 * tau3 - 15 * tau4 + 6 * tau5;
      sd = (30 * tau2 - 60 * tau3 + 30 * tau4) / T;
      sdd = (60 * tau - 180 * tau2 + 120 * tau3) / (T * T);
    }
    const KDL::Vector dp = target_.p - start_.p;
    x.p = start_.p + dp * s;

    // Rotazione relativa espressa nel frame base: R_target = R_rel * R_start
    const KDL::Rotation R_rel = target_.M * start_.M.Inverse();
    KDL::Vector axis;
    const double angle = R_rel.GetRotAngle(axis);
    x.M = KDL::Rotation::Rot2(axis, angle * s) * start_.M;

    v.vel = dp * sd;
    v.rot = axis * (angle * sd);
    a.vel = dp * sdd;
    a.rot = axis * (angle * sdd);
  }

  static double angleBetween(const KDL::Rotation & a, const KDL::Rotation & b)
  {
    KDL::Vector axis;
    return (b * a.Inverse()).GetRotAngle(axis);
  }

  // e_o = eta_e*eps_d - eta_d*eps_e - eps_d x eps_e  (= parte vettoriale di Q_d * Q_e^-1)
  static Eigen::Vector3d orientationError(const KDL::Rotation & Rd, const KDL::Rotation & Re)
  {
    double x, y, z, w;
    Rd.GetQuaternion(x, y, z, w);
    Eigen::Quaterniond qd(w, x, y, z);
    Re.GetQuaternion(x, y, z, w);
    Eigen::Quaterniond qe(w, x, y, z);
    if (qd.dot(qe) < 0.0) {qe.coeffs() *= -1.0;}  // stesso emisfero
    const Eigen::Quaterniond qerr = qd * qe.inverse();
    return qerr.vec();
  }

  static Eigen::Vector3d toEigen(const KDL::Vector & v) {return {v.x(), v.y(), v.z()};}
  static Vector6d twistToEigen(const KDL::Twist & t)
  {
    Vector6d r;
    r << t.vel.x(), t.vel.y(), t.vel.z(), t.rot.x(), t.rot.y(), t.rot.z();
    return r;
  }

  KDL::Chain chain_;
  ClikParams params_;
  unsigned int n_;
  KDL::ChainFkSolverPos_recursive fk_;
  KDL::ChainJntToJacSolver jac_solver_;
  KDL::ChainJntToJacDotSolver jacdot_solver_;

  KDL::JntArray q_, qd_, qdd_;
  KDL::Jacobian jac_;
  KDL::Frame x_, desired_, start_, target_;
  Vector6d error_ = Vector6d::Zero();
  double manipulability_ = 0.0;
  double duration_ = 0.0, t_ = 0.0;
  bool initialized_ = false;
};

}  // namespace agricultural_robot