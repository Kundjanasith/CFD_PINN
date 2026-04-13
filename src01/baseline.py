import pandas as pd
import tf_keras as keras
import numpy as np
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tqdm

def load_and_combine_data(file_100, file_060, file_030):
    print("--- 🔄 กำลังเตรียมและรวมข้อมูล Parametrized PINN ---")
    
    # 1. ตรวจสอบว่าไฟล์ทั้ง 3 มีอยู่จริง
    files = [file_100, file_060, file_030]
    for f in files:
        if not os.path.exists(f):
            print(f"❌ Error: หาไฟล์ '{f}' ไม่เจอ กรุณาตรวจสอบชื่อไฟล์และโฟลเดอร์ครับ")
            return None

    # 2. ฟังก์ชันย่อยสำหรับโหลดและทำความสะอาดไฟล์เดี่ยว
    def process_single_file(filepath, speed_value):
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower() # จัดการชื่อคอลัมน์
        df = df.dropna() # ลบค่าว่าง
        
        # คัดมาเฉพาะคอลัมน์ที่ใช้
        cols = ['x-coordinate', 'y-coordinate', 'z-coordinate', 
                'x-velocity', 'y-velocity', 'z-velocity', 'pressure']
        df = df[cols].copy()
        
        # *** หัวใจสำคัญ: เพิ่มคอลัมน์ความเร็วพัดลม (fan_speed) ***
        df['fan_speed'] = speed_value
        return df

    # 3. โหลดและใส่ค่า Parameter ให้แต่ละเคส
    print(f"กำลังโหลดเคส 100% จาก: {file_100} ...")
    df_100 = process_single_file(file_100, speed_value=1.0)
    
    print(f"กำลังโหลดเคส 60%  จาก: {file_060} ...")
    df_060 = process_single_file(file_060, speed_value=0.6)
    
    print(f"กำลังโหลดเคส 30%  จาก: {file_030} ...")
    df_030 = process_single_file(file_030, speed_value=0.3)

    # 4. รวมข้อมูลทั้งหมดเข้าด้วยกัน (Concatenate)
    df_combined = pd.concat([df_100, df_060, df_030], ignore_index=True)
    
    # สุ่มข้อมูล (Shuffle) เพื่อไม่ให้โมเดลจำแพทเทิร์นแบบเรียงเคส
    df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    # =========================================================
    print("\n💾 กำลังบันทึกไฟล์ข้อมูลที่รวมแล้ว...")
    save_filename = 'combined_pinn_data.csv'
    df_combined.to_csv(save_filename, index=False)
    print(f"✔️ บันทึกไฟล์ '{save_filename}' สำเร็จ!")
    # =========================================================
    
    print(f"\n✔️ รวมข้อมูลสำเร็จ! จำนวนจุดข้อมูลรวมทั้งหมด: {len(df_combined):,} จุด")

    # 5. แยก Input (4 ตัวแปร) และ Output (4 ตัวแปร)
    # Input: x, y, z, fan_speed
    X_raw = df_combined[['x-coordinate', 'y-coordinate', 'z-coordinate', 'fan_speed']].values.astype(np.float32)
    
    # Output: u, v, w, p
    U_raw = df_combined['x-velocity'].values.astype(np.float32).reshape(-1, 1)
    V_raw = df_combined['y-velocity'].values.astype(np.float32).reshape(-1, 1)
    W_raw = df_combined['z-velocity'].values.astype(np.float32).reshape(-1, 1)
    P_raw = df_combined['pressure'].values.astype(np.float32).reshape(-1, 1)

    # 6. Normalization (ปรับสเกล 0 ถึง 1) สำหรับ Input
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    X_range = X_max - X_min
    X_range = np.where(X_range == 0, 1.0, X_range) # ป้องกันการหารด้วยศูนย์
    
    X_norm = (X_raw - X_min) / X_range

    print("\n--- 📊 สรุปตัวแปรที่พร้อมใช้ (Shape) ---")
    print(f"Input Matrix (X_norm): {X_norm.shape} -> [x, y, z, fan_speed]")
    print(f"Output Matrices (U, V, W, P): แต่ละตัวมีขนาด {U_raw.shape}")
    
    return X_norm, U_raw, V_raw, W_raw, P_raw, X_min, X_range

# ==========================================
# วิธีเรียกใช้งาน (เปลี่ยนชื่อไฟล์ตรงนี้)
# ==========================================
file_1 = '../Train/fluent_data_100.csv'  # ไฟล์เคส 100%
file_2 = '../Train/fluent_data_060.csv'  # ไฟล์เคส 60%
file_3 = '../Train/fluent_data_030.csv'  # ไฟล์เคส 30%

data = load_and_combine_data(file_1, file_2, file_3)

if data is not None:
    X_train_norm, u_train, v_train, w_train, p_train, X_min_val, X_range_val = data

# print("\n✅ ข้อมูลพร้อมสำหรับการฝึกโมเดล PINN แล้วครับ!")
# --- 📊 สรุปตัวแปรที่พร้อมใช้ (Shape) ---
# Input Matrix (X_norm): (730365, 4) -> [x, y, z, fan_speed]
# Output Matrices (U, V, W, P): แต่ละตัวมีขนาด (730365, 1)

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

# ==============================================================================
# 1. โหลดข้อมูลรวม (Combined Data) และเตรียม Dataset
# ==============================================================================
filepath = 'combined_pinn_data.csv'

print(f"กำลังโหลดข้อมูลจาก: {filepath} ...")
if not os.path.exists(filepath):
    raise FileNotFoundError(f"❌ ไม่พบไฟล์ {filepath} กรุณาตรวจสอบชื่อไฟล์ครับ")

df = pd.read_csv(filepath)

# แยก Input (4 ตัวแปร: x, y, z, fan_speed) และ Output
X_raw = df[['x-coordinate', 'y-coordinate', 'z-coordinate', 'fan_speed']].values.astype(np.float32)
U_raw = df['x-velocity'].values.astype(np.float32).reshape(-1, 1)
V_raw = df['y-velocity'].values.astype(np.float32).reshape(-1, 1)
W_raw = df['z-velocity'].values.astype(np.float32).reshape(-1, 1)

# Normalization (ปรับสเกล 0-1)
X_min = tf.constant(X_raw.min(axis=0), dtype=tf.float32)
X_max = tf.constant(X_raw.max(axis=0), dtype=tf.float32)
X_range = X_max - X_min
X_range = tf.where(X_range == 0, tf.ones_like(X_range), X_range)

X_train_norm = (X_raw - X_min.numpy()) / X_range.numpy()

# สร้าง Dataset Pipeline เร่งความเร็ว
BATCH_SIZE = 1024  
dataset = tf.data.Dataset.from_tensor_slices(
    (X_train_norm, U_raw, V_raw, W_raw)
).shuffle(buffer_size=50000).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

print(f"✔️ โหลดข้อมูลสำเร็จ! Input Shape: {X_train_norm.shape} | Batch Size: {BATCH_SIZE}")

# ==============================================================================
# 2. สร้างโครงสร้างโมเดลขั้นสูง (เพิ่มระบบ Fourier Features)
# ==============================================================================
def build_fourier_model():
    # ใช้ Functional API เพื่อแยกสายข้อมูล
    inputs = tf.keras.Input(shape=(4,))
    
    # แยกพิกัด x, y, z (3 ตัวแรก) และ fan_speed (ตัวสุดท้าย)
    xyz = inputs[:, 0:3]
    fan = inputs[:, 3:4]
    
    # 🔥 สร้าง Fourier Features ให้กับพิกัด x, y, z
    freqs = [1.0, 2.0, 4.0, 8.0] # ระดับความถี่ (ให้ AI เห็นภาพละเอียดขึ้น 4 ระดับ)
    features = [xyz]
    for freq in freqs:
        features.append(tf.math.sin(np.pi * freq * xyz))
        features.append(tf.math.cos(np.pi * freq * xyz))
        # features.append(keras.ops.sin(np.pi * freq * xyz))
        # features.append(keras.ops.cos(np.pi * freq * xyz))
        
    # นำพิกัดที่แปลงร่างแล้ว มาประกอบร่างรวมกับ fan_speed กลับเหมือนเดิม
    x = tf.concat(features + [fan], axis=-1)
    
    # ส่งเข้าสมองกล (Hidden Layers) แบบ Swish
    x = tf.keras.layers.Dense(128, activation='swish')(x)
    x = tf.keras.layers.Dense(128, activation='swish')(x)
    x = tf.keras.layers.Dense(128, activation='swish')(x)
    x = tf.keras.layers.Dense(128, activation='swish')(x)
    x = tf.keras.layers.Dense(64, activation='swish')(x)
    
    # Output Layer
    outputs = tf.keras.layers.Dense(4)(x) # u, v, w, p
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = build_fourier_model()

# ระบบลด Learning Rate 
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002,   # ดันเริ่มต้นให้สูงนิดนึงเพื่อกระชากกราฟ
    decay_steps=50000,             
    decay_rate=0.90,
    staircase=False
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

print("Initializing model...")
_ = model(tf.zeros((1, 4))) 
print("Model initialized with Fourier Features!")

# -------------------------------------------------------------
# 🔄 ระบบจัดการ Checkpoint (ล้างความจำเดิมอัตโนมัติ)
# -------------------------------------------------------------
checkpoint_path = 'parametrized_pinn_model.weights.h5'

# 🔥 บังคับลบทิ้ง 100% เพราะเราเปลี่ยนโครงสร้าง Input ใหม่ สมองเดิมใช้ไม่ได้แล้ว
if os.path.exists(checkpoint_path):
    print("\n⚠️ โครงสร้างโมเดลเปลี่ยนไป: กำลังลบความจำเดิมทิ้ง...")
    os.remove(checkpoint_path)
    print("🗑️ ลบไฟล์เรียบร้อยแล้ว! จะเริ่มเทรนใหม่ตั้งแต่ 0")
else:
    print(f"\n✨ จะเป็นการเริ่มเทรนโมเดลใหม่ตั้งแต่ต้น...")

# ==============================================================================
# 3. Physics & Loss Function
# ==============================================================================
rho = 1.225
nu = 1.5e-5

@tf.function
def train_step(X_batch_norm, u_true, v_true, w_true):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X_batch_norm)
        
        preds = model(X_batch_norm)
        u, v, w, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3], preds[:, 3:4]
        
        du_dX_norm = tape.gradient(u, X_batch_norm)
        dv_dX_norm = tape.gradient(v, X_batch_norm)
        dw_dX_norm = tape.gradient(w, X_batch_norm)
        dp_dX_norm = tape.gradient(p, X_batch_norm)
        
        inv_range_x, inv_range_y, inv_range_z = 1.0/X_range[0], 1.0/X_range[1], 1.0/X_range[2]
        
        u_x = du_dX_norm[:, 0:1] * inv_range_x
        u_y = du_dX_norm[:, 1:2] * inv_range_y
        u_z = du_dX_norm[:, 2:3] * inv_range_z
        v_x = dv_dX_norm[:, 0:1] * inv_range_x
        v_y = dv_dX_norm[:, 1:2] * inv_range_y
        v_z = dv_dX_norm[:, 2:3] * inv_range_z
        w_x = dw_dX_norm[:, 0:1] * inv_range_x
        w_y = dw_dX_norm[:, 1:2] * inv_range_y
        w_z = dw_dX_norm[:, 2:3] * inv_range_z
        p_x = dp_dX_norm[:, 0:1] * inv_range_x
        p_y = dp_dX_norm[:, 1:2] * inv_range_y
        p_z = dp_dX_norm[:, 2:3] * inv_range_z

        u_xx = tape.gradient(du_dX_norm[:, 0:1], X_batch_norm)[:, 0:1] * (inv_range_x**2)
        u_yy = tape.gradient(du_dX_norm[:, 1:2], X_batch_norm)[:, 1:2] * (inv_range_y**2)
        u_zz = tape.gradient(du_dX_norm[:, 2:3], X_batch_norm)[:, 2:3] * (inv_range_z**2)
        v_xx = tape.gradient(dv_dX_norm[:, 0:1], X_batch_norm)[:, 0:1] * (inv_range_x**2)
        v_yy = tape.gradient(dv_dX_norm[:, 1:2], X_batch_norm)[:, 1:2] * (inv_range_y**2)
        v_zz = tape.gradient(dv_dX_norm[:, 2:3], X_batch_norm)[:, 2:3] * (inv_range_z**2)
        w_xx = tape.gradient(dw_dX_norm[:, 0:1], X_batch_norm)[:, 0:1] * (inv_range_x**2)
        w_yy = tape.gradient(dw_dX_norm[:, 1:2], X_batch_norm)[:, 1:2] * (inv_range_y**2)
        w_zz = tape.gradient(dw_dX_norm[:, 2:3], X_batch_norm)[:, 2:3] * (inv_range_z**2)

        res_continuity = u_x + v_y + w_z
        res_ns_x = (u*u_x + v*u_y + w*u_z) + (1/rho)*p_x - nu*(u_xx + u_yy + u_zz)
        res_ns_y = (u*v_x + v*v_y + w*v_z) + (1/rho)*p_y - nu*(v_xx + v_yy + v_zz)
        res_ns_z = (u*w_x + v*w_y + w*w_z) + (1/rho)*p_z - nu*(w_xx + w_yy + w_zz)
        
        loss_data = tf.reduce_mean(tf.square(u_true - u)) + \
                    tf.reduce_mean(tf.square(v_true - v)) + \
                    tf.reduce_mean(tf.square(w_true - w))
        
        loss_physics = tf.reduce_mean(tf.square(res_continuity)) + \
                       tf.reduce_mean(tf.square(res_ns_x)) + \
                       tf.reduce_mean(tf.square(res_ns_y)) + \
                       tf.reduce_mean(tf.square(res_ns_z))
        
        # 🔥 ปรับสมดุลใหม่แบบ Moderate Forcing (ให้ความสำคัญ Data มากขึ้นนิดหน่อย)
        weight_data = 3.0  
        weight_physics = 0.5 
        total_loss = (weight_data * loss_data) + (weight_physics * loss_physics)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    del tape 
    
    return total_loss, loss_data, loss_physics

# ==============================================================================
# 4. Training Loop 
# ==============================================================================
INITIAL_EPOCH = 0       
ADDITIONAL_EPOCHS = 3000   
TOTAL_EPOCHS = INITIAL_EPOCH + ADDITIONAL_EPOCHS

# PRINT_FREQ = 10     
# SAVE_FREQ = 100     
PRINT_FREQ = 1     
SAVE_FREQ = 1    

print(f"\n🚀 เริ่มต้นฝึกสอนจากรอบที่ {INITIAL_EPOCH} ไปจนถึงรอบที่ {TOTAL_EPOCHS}")
start_time = time.time()

loss_total_history = []
loss_data_history = []
loss_phys_history = []

try:
    for epoch in tqdm.tqdm(range(INITIAL_EPOCH, TOTAL_EPOCHS)):
        epoch_loss_total = 0
        epoch_loss_data = 0
        epoch_loss_phys = 0
        steps = 0
        
        for X_batch, u_b, v_b, w_b in dataset:
            l_total, l_data, l_phys = train_step(X_batch, u_b, v_b, w_b)
            epoch_loss_total += l_total
            epoch_loss_data += l_data
            epoch_loss_phys += l_phys
            steps += 1
        
        avg_loss_total = epoch_loss_total / steps
        avg_loss_data = epoch_loss_data / steps
        avg_loss_phys = epoch_loss_phys / steps
        
        loss_total_history.append(avg_loss_total.numpy())
        loss_data_history.append(avg_loss_data.numpy())
        loss_phys_history.append(avg_loss_phys.numpy())
        
        if (epoch + 1) % PRINT_FREQ == 0:
            # current_lr = optimizer._decayed_lr(tf.float32).numpy()
            current_lr = optimizer.learning_rate.numpy()
            print(f"Epoch {epoch+1:4d}/{TOTAL_EPOCHS} | LR: {current_lr:.6f} | Total: {avg_loss_total:.6f} | Data: {avg_loss_data:.6f} | Phys: {avg_loss_phys:.6f}")

        if (epoch + 1) % SAVE_FREQ == 0:
            # model.save_weights(checkpoint_path)
            model.save_weights('models/model_checkpoint_epoch{}.weights.h5'.format(epoch+1))

except KeyboardInterrupt:
    print("\n\n⚠️ คุณได้กดหยุดการทำงาน (Interrupted by User)")
    print("กำลังบันทึกน้ำหนักล่าสุด...")

# ==============================================================================
# 5. สรุปผลและวาดกราฟ (Plotting)
# ==============================================================================
model.save_weights(checkpoint_path)
print(f"\n✅ อัปเดตไฟล์โมเดลล่าสุดเรียบร้อยที่: {checkpoint_path}")
print(f"ใช้เวลาเทรนไป: {(time.time() - start_time) / 60:.2f} นาที")

plt.figure(figsize=(10, 6))
plt.plot(range(INITIAL_EPOCH, INITIAL_EPOCH + len(loss_total_history)), loss_total_history, label='Total Loss', color='black', linewidth=2)
plt.plot(range(INITIAL_EPOCH, INITIAL_EPOCH + len(loss_data_history)), loss_data_history, label='Data Loss', color='blue', linestyle='--', alpha=0.8)
plt.plot(range(INITIAL_EPOCH, INITIAL_EPOCH + len(loss_phys_history)), loss_phys_history, label='Physics Loss', color='red', linestyle='-.', alpha=0.8)

plt.yscale('log') 
plt.title('Training Loss Breakdown (Fourier Features Enabled)')
plt.xlabel('Epochs')
plt.ylabel('Loss Value (Log Scale)')
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()

# -------------------------------------------------------------
# 🔥 เพิ่มระบบเซฟรูปกราฟ (PNG และ EPS) แบบไม่เซฟทับรูปเดิม
# -------------------------------------------------------------
# ใช้ตัวแปร epoch ที่ได้จากการรัน มาตั้งเป็นชื่อไฟล์ด้วยเลย
filename_base = f'training_loss_fourier_epoch{epoch+1}'

# หรือถ้าอยากให้มีวันที่และเวลาต่อท้ายด้วย (ป้องกันไฟล์ซ้ำ 100%) ใช้โค้ดนี้แทนได้ครับ:
# import datetime
# current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# filename_base = f'training_loss_fourier_epoch{epoch+1}_{current_time}'

# บันทึกเป็น PNG (ความละเอียดสูง 300 DPI)
plt.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight', facecolor='white')

# บันทึกเป็น EPS (สำหรับตีพิมพ์เปเปอร์วิจัย)
plt.savefig(f'{filename_base}.eps', format='eps', bbox_inches='tight', facecolor='white')