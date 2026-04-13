import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 0. เช็คก่อนว่าในข้อมูลเรามีพัดลมความเร็วเท่าไหร่บ้าง
available_fans = np.unique(X_raw[:, 3])
print(f"📌 ความเร็วพัดลมที่มีในชุดข้อมูลคือ: {available_fans}")

# 1. เลือกความเร็วพัดลม (เลือกจากตัวเลขที่ปริ้นท์ออกมาด้านบน)
# สมมติว่าในระบบมี 100.0 เราก็ใส่ 100.0 แต่ถ้ามี 1.0 ก็ใส่ 1.0 ครับ
fan_target = available_fans[-1] # ดึงค่าสูงสุดมาใช้ทดสอบอัตโนมัติ

print(f"กำลังดึงข้อมูล Node ที่มีความเร็วพัดลม = {fan_target} ...")

# ใช้ np.isclose แทน == เพื่อป้องกันปัญหาทศนิยมเหลื่อมล้ำ
idx = np.where(np.isclose(X_raw[:, 3], fan_target))[0]

# เช็คความปลอดภัย ถ้าหาไม่เจอให้หยุดการทำงาน
if len(idx) == 0:
    print(f"❌ หาข้อมูลพัดลม {fan_target} ไม่เจอ! โปรดตรวจสอบตัวเลขอีกครั้ง")
else:
    print(f"✔️ พบข้อมูลทั้งหมด {len(idx)} Nodes")

    # ดึงข้อมูลพิกัดและความเร็วลมจริง
    X_nodes = X_raw[idx]
    u_true = U_raw[idx].flatten()
    v_true = V_raw[idx].flatten()
    w_true = W_raw[idx].flatten()
    vel_true = np.sqrt(u_true**2 + v_true**2 + w_true**2)

    # 2. ให้ AI ทำนายผล
    X_nodes_norm = (X_nodes - X_min.numpy()) / X_range.numpy()
    preds = model.predict(tf.cast(X_nodes_norm, tf.float32), batch_size=2048, verbose=0)

    u_pred = preds[:, 0]
    v_pred = preds[:, 1]
    w_pred = preds[:, 2]
    vel_pred = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)

    # 3. พลอตกราฟ Scatter (True vs Predicted)
    plt.figure(figsize=(8, 8))
    plt.scatter(vel_true, vel_pred, alpha=0.3, color='blue', edgecolors='none', label='Nodes')

    min_val = min(np.min(vel_true), np.min(vel_pred))
    max_val = max(np.max(vel_true), np.max(vel_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (100% Match)')

    plt.title(f'Node-by-Node Comparison (Fan Speed = {fan_target})', fontsize=14, fontweight='bold')
    plt.xlabel('CFD Actual Velocity (m/s)', fontsize=12)
    plt.ylabel('AI Predicted Velocity (m/s)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()