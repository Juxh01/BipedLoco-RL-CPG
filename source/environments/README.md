# MyoAssistLeg (baseline) – Action and Observation Spaces

This document describes the flat ActionSpace and ObservationSpace emitted by the unwrapped environment implemented in `myoassist_leg_base.py`. It reflects:
- Model: `models/26muscle_3D/myoLeg26_BASELINE.xml`
- Config: `configs/example.yaml`

The environment constructs the observation vector by concatenating, in order:
[qpos, qvel, act, sensor, target_velocity].
An auxiliary time scalar is available in `get_obs_dict(sim)["time"]` but is not part of the returned observation vector.

## ActionSpace

- Type: Box(low=0.0, high=1.0, shape=(26,), dtype=float32)
- Semantics: Continuous muscle excitations in [0, 1], mapped to muscle dynamics by MuJoCo’s muscle actuator model.
- Ordering (exact actuator order from the XML; indices in the action vector):

0. abd_r
1. add_r
2. hamstrings_r
3. bifemsh_r
4. edl_r
5. fdl_r
6. glutmax_r
7. iliopsoas_r
8. rectfem_r
9. vasti_r
10. gastroc_r
11. soleus_r
12. tibant_r
13. abd_l
14. add_l
15. hamstrings_l
16. bifemsh_l
17. edl_l
18. fdl_l
19. glutmax_l
20. iliopsoas_l
21. rectfem_l
22. vasti_l
23. gastroc_l
24. soleus_l
25. tibant_l

Notes:
- The action order equals the actuator declaration order in the model XML.
- Units: dimensionless (normalized) excitations; resulting torques/forces arise from the muscle model.

## ObservationSpace (flat)

- Type: Box(shape=(75,), dtype=float32) for the current config
- Composition and index ranges:

1) Joint positions qpos (18 scalars): indices [0 … 17]
- ankle_angle_l
- ankle_angle_r
- hip_flexion_l
- hip_flexion_r
- knee_angle_l
- knee_angle_r
- pelvis_tilt
- pelvis_ty
- mtp_angle_l
- mtp_angle_r
- hip_adduction_l
- hip_adduction_r
- hip_rotation_l
- hip_rotation_r
- pelvis_tx
- pelvis_tz
- pelvis_list
- pelvis_rotation

2) Joint velocities qvel (18 scalars): indices [18 … 35]
- ankle_angle_l
- ankle_angle_r
- hip_flexion_l
- hip_flexion_r
- knee_angle_l
- knee_angle_r
- pelvis_tilt
- pelvis_tx
- pelvis_ty
- mtp_angle_l
- mtp_angle_r
- hip_adduction_l
- hip_adduction_r
- hip_rotation_l
- hip_rotation_r
- pelvis_tz
- pelvis_list
- pelvis_rotation

3) Muscle activations act (26 scalars): indices [36 … 61]
- Same order as the ActionSpace:
  abd_r, add_r, hamstrings_r, bifemsh_r, edl_r, fdl_r, glutmax_r, iliopsoas_r, rectfem_r, vasti_r, gastroc_r, soleus_r, tibant_r, abd_l, add_l, hamstrings_l, bifemsh_l, edl_l, fdl_l, glutmax_l, iliopsoas_l, rectfem_l, vasti_l, gastroc_l, soleus_l, tibant_l.

4) Sensors (12 scalars): indices [62 … 73]
Order (from config’s `observation_sensor_keys`) and meaning:
- r_foot        (touch, normalized by body weight)
- l_foot        (touch, normalized by body weight)
- r_toes        (touch, normalized by body weight)
- l_toes        (touch, normalized by body weight)
- r_knee_sensor (jointlimitfrc, hinge-limit reaction torque, raw)
- l_knee_sensor (jointlimitfrc, hinge-limit reaction torque, raw)
- r_hip_sensor  (jointlimitfrc, hinge-limit reaction torque, raw)
- l_hip_sensor  (jointlimitfrc, hinge-limit reaction torque, raw)
- r_ankle_sensor (jointlimitfrc, hinge-limit reaction torque, raw)
- l_ankle_sensor (jointlimitfrc, hinge-limit reaction torque, raw)
- r_mtp_sensor   (jointlimitfrc, hinge-limit reaction torque, raw)
- l_mtp_sensor   (jointlimitfrc, hinge-limit reaction torque, raw)

5) Target forward velocity (1 scalar): index [74]
- target_velocity (m/s), the task’s current setpoint according to the selected velocity mode.

Auxiliary (not in the flat observation array):
- time: sim time in seconds, accessible via `get_obs_dict(sim)["time"]`.

### Units and normalization

- Angles: radians; translations: meters.
- Angular velocities: rad/s; linear velocities: m/s.
- Foot/toe touch sensors: normalized by total body weight (sum(body_mass)*9.81) → dimensionless.
- jointlimitfrc sensors: raw hinge-limit reaction torques (N·m). They are not normalized in the observation function.
- Muscle activations (act): current internal muscle activation states [0, 1].

## What “jointlimitfrc” means

MuJoCo’s `jointlimitfrc` reports the solver’s reaction at joint limits:
- Hinge: reaction torque [N·m] about the joint axis.
- Slide: reaction force [N] along the slide axis.
It is zero unless a limit is active; the sign indicates the active side.

In this model, all listed limit sensors are for hinge joints and thus report torques:
- r_knee_sensor, l_knee_sensor
- r_hip_sensor, l_hip_sensor
- r_ankle_sensor, l_ankle_sensor
- r_mtp_sensor, l_mtp_sensor

## Extending the ObservationSpace via configuration

You can extend or reduce qpos/qvel/sensor by editing the lists in your config:
- env.env_params.observation_joint_pos_keys
- env.env_params.observation_joint_vel_keys
- env.env_params.observation_sensor_keys

The observation vector adjusts automatically. Use only names that exist in the model XML.

Valid joint names in the baseline model (selection):

- Pelvis (floating base coordinates):
  - pelvis_tx, pelvis_ty, pelvis_tz
  - pelvis_tilt, pelvis_list, pelvis_rotation

- Lower limbs:
  - Right:  hip_flexion_r, hip_adduction_r, hip_rotation_r, knee_angle_r, ankle_angle_r, mtp_angle_r
  - Left:   hip_flexion_l, hip_adduction_l, hip_rotation_l, knee_angle_l, ankle_angle_l, mtp_angle_l
  - Knee auxiliary slide coordinates (planar knee model, advanced use):
    - knee_r_translation1, knee_r_translation2
    - knee_l_translation1, knee_l_translation2

- Non-actuated upper limbs (optional to observe):
  - Right shoulder/elbow/wrist:
    r_shoulder_abd, r_shoulder_rot, r_shoulder_flex, r_elbow_flex,
    r_wrist_rot, r_wrist_flex, r_wrist_dev
  - Left shoulder/elbow/wrist:
    l_shoulder_abd, l_shoulder_rot, l_shoulder_flex, l_elbow_flex,
    l_wrist_rot, l_wrist_flex, l_wrist_dev

- Advanced (via points for muscle paths; kinematic helpers). These exist but are typically not needed in RL observations:
  - hamstrings_r_semimem_r-P2_x, hamstrings_r_semimem_r-P2_y
  - hamstrings_l_semimem_l-P2_x, hamstrings_l_semimem_l-P2_y
  - rect_fem_r_rect_fem_r-P3_x, rect_fem_r_rect_fem_r-P3_y
  - rect_fem_l_rect_fem_l-P3_x, rect_fem_l_rect_fem_l-P3_y
  - vasti_r_vas_int_r-P4_x, vasti_r_vas_int_r-P4_y
  - vasti_l_vas_int_l-P4_x, vasti_l_vas_int_l-P4_y
  - gastroc_r_med_gas_r-P2_x, gastroc_r_med_gas_r-P2_y, gastroc_r_med_gas_r-P2_z
  - gastroc_l_med_gas_l-P2_x, gastroc_l_med_gas_l-P2_y, gastroc_l_med_gas_l-P2_z
  - iliopsoas_r_psoas_r-P3_x, iliopsoas_r_psoas_r-P3_y, iliopsoas_r_psoas_r-P3_z
  - iliopsoas_l_psoas_l-P3_x, iliopsoas_l_psoas_l-P3_y, iliopsoas_l_psoas_l-P3_z

Valid sensor names in the baseline model (exactly those defined in the XML):
- Touch (weight-normalized): r_foot, r_toes, l_foot, l_toes
- Joint-limit reactions (raw): r_knee_sensor, l_knee_sensor, r_hip_sensor, l_hip_sensor, r_ankle_sensor, l_ankle_sensor, r_mtp_sensor, l_mtp_sensor

Notes:
- All listed sensors are scalar in this model.
- For any sensor not included above, you would need to add a definition in the XML.

## Dimension accounting (current config)

Let:
- nqpos = len(observation_joint_pos_keys) = 18
- nqvel = len(observation_joint_vel_keys) = 18
- nact  = number of muscle actuators = 26
- nsens = sum of per-sensor dimensions = 12
- ntv   = 1 (target velocity)

Total observation dimension = nqpos + nqvel + nact + nsens + ntv = 18 + 18 + 26 + 12 + 1 = 75.

## Practical remarks

- The returned observation is a flat vector; the ordering is strictly:
  qpos → qvel → act → sensor → target_velocity.
- The internal `get_obs_dict(sim)` exposes the same content split by blocks and includes a `'time'` field.
- The reward uses `joint_limit_sensor_keys` to compute a joint-constraint force/torque penalty; those sensor values are not normalized in observations.