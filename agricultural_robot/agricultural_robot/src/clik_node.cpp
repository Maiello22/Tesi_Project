// Nodo ROS 2 per il CLIK del secondo ordine dell'UR5e.
//
// Ingressi
//   /robot_description  (std_msgs/String, transient local)  -> catena KDL base_link -> tip_link
//   /joint_states       (sensor_msgs/JointState)            -> stato iniziale e retroazione
//   /clik/target_pose   (geometry_msgs/PoseStamped)         -> obiettivo, in qualunque frame TF
//
// Uscite
//   /forward_position_controller/commands  oppure  /forward_velocity_controller/commands
//   /clik/error         (std_msgs/Float64MultiArray)  [ex ey ez eox eoy eoz |e_p| |e_o| w]
//   /clik/current_pose  (geometry_msgs/PoseStamped)   posa raggiunta (FK dello stato CLIK)
//   /clik/desired_pose  (geometry_msgs/PoseStamped)   riferimento quintico istantaneo

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "agricultural_robot/clik_core.hpp"

using namespace std::chrono_literals;
using agricultural_robot::ClikController;
using agricultural_robot::ClikParams;

class ClikNode : public rclcpp::Node
{
public:
  ClikNode()
  : Node("clik_node")
  {
    base_link_ = declare_parameter<std::string>("base_link", "base_link");
    tip_link_ = declare_parameter<std::string>("tip_link", "zed_left_camera_frame_optical");
    rate_ = declare_parameter<double>("control_rate", 200.0);
    command_interface_ = declare_parameter<std::string>("command_interface", "position");
    vel_fb_gain_ = declare_parameter<double>("velocity_feedback_gain", 5.0);
    controller_joints_ = declare_parameter<std::vector<std::string>>(
      "controller_joints",
      {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"});

    p_.kp_pos = declare_parameter<double>("kp_pos", p_.kp_pos);
    p_.kp_ori = declare_parameter<double>("kp_ori", p_.kp_ori);
    p_.kd_pos = declare_parameter<double>("kd_pos", p_.kd_pos);
    p_.kd_ori = declare_parameter<double>("kd_ori", p_.kd_ori);
    p_.lambda_max = declare_parameter<double>("lambda_max", p_.lambda_max);
    p_.w_threshold = declare_parameter<double>("manipulability_threshold", p_.w_threshold);
    p_.max_joint_vel = declare_parameter<double>("max_joint_vel", p_.max_joint_vel);
    p_.max_joint_acc = declare_parameter<double>("max_joint_acc", p_.max_joint_acc);
    p_.max_lin_vel = declare_parameter<double>("max_lin_vel", p_.max_lin_vel);
    p_.max_ang_vel = declare_parameter<double>("max_ang_vel", p_.max_ang_vel);
    p_.min_duration = declare_parameter<double>("min_duration", p_.min_duration);
    p_.max_pos_error = declare_parameter<double>("max_pos_error", p_.max_pos_error);
    p_.max_ori_error = declare_parameter<double>("max_ori_error", p_.max_ori_error);
    p_.kd_null = declare_parameter<double>("kd_null", p_.kd_null);
    shoulder_height_ = declare_parameter<double>("shoulder_height", 0.1625);
    max_reach_ = declare_parameter<double>("max_reach", 0.85);

    const std::string cmd_topic = command_interface_ == "velocity" ?
      "/forward_velocity_controller/commands" : "/forward_position_controller/commands";
    cmd_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(cmd_topic, 10);
    err_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/clik/error", 10);
    cur_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/clik/current_pose", 10);
    des_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/clik/desired_pose", 10);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    desc_sub_ = create_subscription<std_msgs::msg::String>(
      "/robot_description", rclcpp::QoS(1).transient_local(),
      [this](std_msgs::msg::String::SharedPtr msg) {onDescription(msg->data);});
    js_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::JointState::SharedPtr msg) {onJointState(*msg);});
    target_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/clik/target_pose", 10,
      [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) {onTarget(*msg);});

    // Timer sul clock del nodo: con use_sim_time segue il tempo di Gazebo
    timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(1.0 / rate_),
      [this]() {controlLoop();});

    RCLCPP_INFO(
      get_logger(), "CLIK 2o ordine: %s -> %s, %.0f Hz, comando in %s su %s",
      base_link_.c_str(), tip_link_.c_str(), rate_, command_interface_.c_str(), cmd_topic.c_str());
  }

private:
  void onDescription(const std::string & urdf)
  {
    if (clik_) {return;}
    KDL::Tree tree;
    if (!kdl_parser::treeFromString(urdf, tree)) {
      RCLCPP_ERROR(get_logger(), "Impossibile costruire l'albero KDL dal robot_description");
      return;
    }
    KDL::Chain chain;
    if (!tree.getChain(base_link_, tip_link_, chain)) {
      RCLCPP_ERROR(get_logger(), "Catena %s -> %s non trovata", base_link_.c_str(), tip_link_.c_str());
      return;
    }
    chain_joints_.clear();
    for (const auto & seg : chain.segments) {
      if (seg.getJoint().getType() != KDL::Joint::None) {
        chain_joints_.push_back(seg.getJoint().getName());
      }
    }
    clik_ = std::make_unique<ClikController>(chain, p_);
    RCLCPP_INFO(get_logger(), "Catena KDL pronta: %u giunti, %u segmenti",
      chain.getNrOfJoints(), chain.getNrOfSegments());
  }

  void onJointState(const sensor_msgs::msg::JointState & msg)
  {
    for (size_t i = 0; i < msg.name.size() && i < msg.position.size(); ++i) {
      joint_pos_[msg.name[i]] = msg.position[i];
    }
    have_js_ = true;
  }

  bool measuredState(Eigen::VectorXd & q) const
  {
    q.resize(chain_joints_.size());
    for (size_t i = 0; i < chain_joints_.size(); ++i) {
      auto it = joint_pos_.find(chain_joints_[i]);
      if (it == joint_pos_.end()) {return false;}
      q(i) = it->second;
    }
    return true;
  }

  void onTarget(const geometry_msgs::msg::PoseStamped & msg)
  {
    if (!clik_ || !clik_->initialized()) {
      RCLCPP_WARN(get_logger(), "Obiettivo ignorato: CLIK non ancora inizializzato");
      return;
    }
    geometry_msgs::msg::PoseStamped in_base;
    try {
      in_base = tf_buffer_->transform(msg, base_link_, tf2::durationFromSec(0.2));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "Obiettivo non trasformabile in %s: %s", base_link_.c_str(), ex.what());
      return;
    }
    const auto & p = in_base.pose.position;
    const auto & o = in_base.pose.orientation;
    KDL::Frame target(KDL::Rotation::Quaternion(o.x, o.y, o.z, o.w), KDL::Vector(p.x, p.y, p.z));

    // Controllo di raggiungibilita': distanza dalla spalla (asse shoulder_lift)
    const double reach = (target.p - KDL::Vector(0.0, 0.0, shoulder_height_)).Norm();
    if (reach > max_reach_) {
      RCLCPP_WARN(get_logger(),
        "Obiettivo rifiutato: a %.3f m dalla spalla, oltre il limite di %.3f m "
        "(braccio troppo disteso / singolarita'). Avvicina il marker o riduci la distanza.",
        reach, max_reach_);
      return;
    }
    clik_->setTarget(target);
    RCLCPP_INFO(get_logger(), "Nuovo obiettivo in %s: [%.3f %.3f %.3f], %.3f m dalla spalla, durata %.2f s",
      base_link_.c_str(), p.x, p.y, p.z, reach, clik_->duration());
  }

  void controlLoop()
  {
    if (!clik_ || !have_js_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
        "In attesa di robot_description e joint_states...");
      return;
    }
    Eigen::VectorXd q_meas;
    if (!measuredState(q_meas)) {return;}

    const rclcpp::Time now = get_clock()->now();
    if (!clik_->initialized()) {
      clik_->reset(q_meas);
      last_time_ = now;
      RCLCPP_INFO(get_logger(), "CLIK inizializzato sulla posa attuale: mantengo la posizione");
      return;
    }

    double dt = (now - last_time_).seconds();
    last_time_ = now;
    if (dt <= 0.0) {return;}               // clock fermo (simulazione in pausa)
    dt = std::min(dt, 5.0 / rate_);        // evita passi enormi dopo una pausa

    if (!clik_->step(dt)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Passo CLIK non valido");
      return;
    }

    publishCommand(q_meas);
    publishDiagnostics(now);
  }

  void publishCommand(const Eigen::VectorXd & q_meas)
  {
    const Eigen::VectorXd q = clik_->q();
    const Eigen::VectorXd qd = clik_->qd();
    std_msgs::msg::Float64MultiArray cmd;
    cmd.data.resize(controller_joints_.size(), 0.0);
    for (size_t j = 0; j < controller_joints_.size(); ++j) {
      for (size_t i = 0; i < chain_joints_.size(); ++i) {
        if (chain_joints_[i] != controller_joints_[j]) {continue;}
        cmd.data[j] = command_interface_ == "velocity" ?
          qd(i) + vel_fb_gain_ * (q(i) - q_meas(i)) :  // velocita' + correzione di deriva
          q(i);
      }
    }
    cmd_pub_->publish(cmd);
  }

  void publishDiagnostics(const rclcpp::Time & stamp)
  {
    const auto & e = clik_->error();
    std_msgs::msg::Float64MultiArray err;
    err.data = {e(0), e(1), e(2), e(3), e(4), e(5),
      e.head<3>().norm(), e.tail<3>().norm(), clik_->manipulability()};
    err_pub_->publish(err);
    cur_pub_->publish(toPose(clik_->currentPose(), stamp));
    des_pub_->publish(toPose(clik_->desiredPose(), stamp));
  }

  geometry_msgs::msg::PoseStamped toPose(const KDL::Frame & f, const rclcpp::Time & stamp) const
  {
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = stamp;
    ps.header.frame_id = base_link_;
    ps.pose.position.x = f.p.x();
    ps.pose.position.y = f.p.y();
    ps.pose.position.z = f.p.z();
    f.M.GetQuaternion(
      ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w);
    return ps;
  }

  // Parametri
  std::string base_link_, tip_link_, command_interface_;
  double rate_, vel_fb_gain_, shoulder_height_, max_reach_;
  std::vector<std::string> controller_joints_;
  ClikParams p_;

  // Stato
  std::unique_ptr<ClikController> clik_;
  std::vector<std::string> chain_joints_;
  std::map<std::string, double> joint_pos_;
  bool have_js_ = false;
  rclcpp::Time last_time_;

  // ROS
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_, err_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr cur_pub_, des_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr desc_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_sub_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ClikNode>());
  rclcpp::shutdown();
  return 0;
}