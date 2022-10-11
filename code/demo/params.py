
params = {
    "tool_reactor_wrist": {
        'py_class_name': 'ToolReactorWrist',
        'py_module_name': 'stretch_tool_share.reactorx_wrist_v1.tool',
        'use_group_sync_read': 1,
        'retry_on_comm_failure': 1,
        'baud':115200,
        'dxl_latency_timer': 64,
        'stow': {
            'arm': 0.0,
            'lift': 0.3,
            'reactor_gripper': 0.0,
            'wrist_pitch': 2.0,
            'wrist_roll': 0.0,
            'wrist_yaw': 3.0
        },
        'devices': {
            'reactor_gripper': {
                'py_class_name': 'ReactorGripper',
                'py_module_name': 'stretch_tool_share.reactorx_wrist_v1.reactor_gripper',
            },
            'wrist_pitch': {
                'py_class_name': 'WristPitch',
                'py_module_name': 'stretch_tool_share.reactorx_wrist_v1.wrist_pitch',
                'ros_py_class_name': 'WristPitchCommandGroup',
                'ros_py_module_name': 'stretch_tool_share.reactorx_wrist_v1.command_groups'
            },
            'wrist_roll': {
                'py_class_name': 'WristRoll',
                'py_module_name': 'stretch_tool_share.reactorx_wrist_v1.wrist_roll',
                'ros_py_class_name': 'WristRollCommandGroup',
                'ros_py_module_name': 'stretch_tool_share.reactorx_wrist_v1.command_groups'
            },
            'wrist_yaw': {
                'py_class_name': 'WristYaw',
                'py_module_name': 'stretch_body.wrist_yaw',
            },
        },
        'collision_models': []
    },

    "reactor_gripper": {
        'flip_encoder_polarity': 0,
        'enable_runstop': 1,
        'gr': 1.0,
        'id': 16,
        'max_voltage_limit': 15,
        'min_voltage_limit': 11,
        'motion': {
            'default': {'accel': 8.0, 'vel': 3.0},
            'fast': {'accel': 15.0, 'vel': 6.0},
            'max': {'accel': 15.0, 'vel': 6.0},
            'slow': {'accel': 4.0, 'vel': 1.0},
            'trajectory_max': {'accel_r': 16.0, 'vel_r': 8.0},
        },
        'pid': [640, 0, 0],
        'pwm_homing': [0, 0],
        'pwm_limit': 885,
        #'range_t': [0, 4096],
        'range_t': [1565, 2869],
        'req_calibration': 0,
        'return_delay_time': 0,
        'stall_backoff': 0.017,
        'stall_max_effort': 10.0,
        'stall_max_time': 1.0,
        'stall_min_vel': 0.1,
        'temperature_limit': 72,
        'usb_name': '/dev/hello-dynamixel-wrist',
        'use_multiturn': 0,
        'zero_t': 1565,
        'baud': 115200,
        'retry_on_comm_failure': 1,
        'disable_torque_on_stop': 1
    },
    "wrist_pitch": {
        'flip_encoder_polarity': 0,
        'enable_runstop': 1,
        'gr': 1.0,
        'id': 14,
        'max_voltage_limit': 15,
        'min_voltage_limit': 11,
        'motion': {
            'default': {'accel': 8.0, 'vel': 3.0},
            'fast': {'accel': 15.0, 'vel': 6.0},
            'max': {'accel': 15.0, 'vel': 6.0},
            'slow': {'accel': 4.0, 'vel': 1.0},
            'trajectory_max': {'accel_r': 16.0, 'vel_r': 8.0},
        },
        'pid': [640, 0, 0],
        'pwm_homing': [0, 0],
        'pwm_limit': 885,
        #'range_t': [650, 2048],
        'range_t': [2052, 3232],
        'req_calibration': 0,
        'return_delay_time': 0,
        'stall_backoff': 0.017,
        'stall_max_effort': 10.0,
        'stall_max_time': 1.0,
        'stall_min_vel': 0.1,
        'temperature_limit': 72,
        'usb_name': '/dev/hello-dynamixel-wrist',
        'use_multiturn': 0,
        'zero_t': 2070,
        'baud': 115200,
        'retry_on_comm_failure': 1,
        'disable_torque_on_stop': 0
    },
    "wrist_roll": {
        'flip_encoder_polarity': 0,
        'enable_runstop': 1,
        'gr': 1.0,
        'id': 15,
        'max_voltage_limit': 15,
        'min_voltage_limit': 11,
        'motion': {
            'default': {'accel': 8.0, 'vel': 3.0},
            'fast': {'accel': 15.0, 'vel': 6.0},
            'max': {'accel': 15.0, 'vel': 6.0},
            'slow': {'accel': 4.0, 'vel': 1.0},
            'trajectory_max': {'accel_r': 16.0, 'vel_r': 8.0},
        },
        'pid': [640, 0, 0],
        'pwm_homing': [0, 0],
        'pwm_limit': 885,
        'range_t': [0, 4095],
        'req_calibration': 0,
        'return_delay_time': 0,
        'stall_backoff': 0.017,
        'stall_max_effort': 10.0,
        'stall_max_time': 1.0,
        'stall_min_vel': 0.1,
        'temperature_limit': 72,
        'usb_name': '/dev/hello-dynamixel-wrist',
        'use_multiturn': 0,
        'zero_t': 2090,
        'baud': 115200,
        'retry_on_comm_failure': 1,
        'disable_torque_on_stop': 0
    }}
