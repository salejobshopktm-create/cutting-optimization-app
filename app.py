import streamlit as st
import pandas as pd
from collections import Counter
from ortools.linear_solver import pywraplp
from fpdf import FPDF

st.set_page_config(page_title="1D Steel Cutting Optimization", layout="wide")

# ชุดโทนสีสำหรับแยกความยาวชิ้นงานตัด (HEX & RGB)
COLOR_PALETTE = [
    ("#1f77b4", (31, 119, 180)),   # น้ำเงิน
    ("#2ca02c", (44, 160, 44)),   # เขียว
    ("#ff7f0e", (255, 127, 14)),  # ส้ม
    ("#9467bd", (148, 103, 189)), # ม่วง
    ("#d62728", (214, 39, 40)),   # แดง
    ("#17becf", (23, 190, 207)),  # ฟ้า
    ("#e377c2", (227, 119, 194)), # ชมพู
    ("#8c564b", (140, 86, 75)),   # น้ำตาล
    ("#bcbd22", (188, 189, 34)),  # เขียวมะนาว
    ("#7f7f7f", (127, 127, 127)), # เทา
]

def get_color_map(unique_cuts):
    color_map = {}
    for idx, cut in enumerate(sorted(unique_cuts)):
        hex_c, rgb_c = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        color_map[cut] = {"hex": hex_c, "rgb": rgb_c}
    return color_map

def optimize_cutting_multi_stock(stock_list, kerf_width, demands):
    items = []
    for length, qty in demands:
        items.extend([int(length)] * int(qty))
    num_items = len(items)
    if num_items == 0:
        return "NO_ITEMS"

    stocks_flat = []
    for length, qty in stock_list:
        stocks_flat.extend([int(length)] * int(qty))
    
    num_stocks = len(stocks_flat)
    if num_stocks == 0:
        return "NO_STOCKS"

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return "NO_SOLVER"

    x = {}
    for i in range(num_stocks):
        for j in range(num_items):
            x[(i, j)] = solver.IntVar(0, 1, f'x_{i}_{j}')

    y = {}
    for i in range(num_stocks):
        y[i] = solver.IntVar(0, 1, f'y_{i}')

    for j in range(num_items):
        solver.Add(sum(x[(i, j)] for i in range(num_stocks)) == 1)

    for i in range(num_stocks):
        stock_len = stocks_flat[i]
        effective_stock = stock_len + kerf_width
        solver.Add(
            sum((items[j] + kerf_width) * x[(i, j)] for j in range(num_items)) <= effective_stock * y[i]
        )

    solver.Minimize(solver.Sum([stocks_flat[i] * y[i] for i in range(num_stocks)]))
    solver.set_time_limit(20000)

    status = solver.Solve()

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raw_results = []
        for i in range(num_stocks):
            if y[i].solution_value() > 0.5:
                pattern = [items[j] for j in range(num_items) if x[(i, j)].solution_value() > 0.5]
                pattern.sort(reverse=True)
                raw_results.append((stocks_flat[i], tuple(pattern)))
        return raw_results
    return "INFEASIBLE"

def generate_pdf(grouped_patterns, stock_summary, demand_summary, kerf_width, color_map):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "1D Cutting Optimization Report", ln=1, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, f"Kerf Width: {kerf_width} mm  |  Total Stock Used: {sum(stock_summary.values())} pcs", ln=1, align="C")
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "1. Stock Material Summary", ln=1)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(60, 7, "Stock Length (mm)", border=1, align="C", fill=True)
    pdf.cell(60, 7, "Quantity Required (pcs)", border=1, align="C", fill=True)
    pdf.ln()
    
    pdf.set_font("Helvetica", "", 9)
    for s_len, qty in sorted(stock_summary.items()):
        pdf.cell(60, 6, f"{s_len:,}", border=1, align="C")
        pdf.cell(60, 6, f"{qty:,} pcs", border=1, align="C")
        pdf.ln()
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. Cut Length Color Legend", ln=1)
    pdf.set_font("Helvetica", "", 9)
    
    for cut in sorted(color_map.keys()):
        r, g, b = color_map[cut]["rgb"]
        pdf.set_fill_color(r, g, b)
        
        if pdf.get_y() > 275:
            pdf.add_page()
            
        pdf.cell(8, 5, "", border=1, fill=True)
        pdf.set_x(25)
        qty = demand_summary.get(cut, 0)
        pdf.cell(0, 5, f"Cut Length: {cut:,} mm  (Total Required: {qty:,} pcs)", ln=1)
        pdf.ln(1)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "3. Grouped Cutting Patterns Layout", ln=1)
    
    max_bar_w = 170.0
    for idx, p in enumerate(grouped_patterns, 1):
        s_len = p['stock_length']
        count = p['count']
        cuts = p['cuts']
        waste = p['waste']
        used = p['used']
        
        if pdf.get_y() > 250:
            pdf.add_page()

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"Pattern #{idx}  |  Stock: {s_len:,} mm  ==>  Repeat Cut: x {count} bars", ln=1)
        
        # --- อัปเดตใหม่: จัดกลุ่มตัวเลขที่ซ้ำกันเป็น "ความยาว x จำนวน" ---
        cut_counts = Counter(cuts)
        cut_text = ", ".join([f"{k} x {v}" for k, v in cut_counts.items()])
        
        pdf.set_font("Helvetica", "", 8)
        # ใช้ multi_cell เพื่อให้ตัวหนังสือปัดบรรทัดให้อัตโนมัติหากยาวเกินหน้ากระดาษ
        pdf.multi_cell(0, 4, f"Used: {used:,} mm | Waste: {waste:,} mm | Cut Pieces: {cut_text}")
        
        start_x = 15.0
        start_y = pdf.get_y() + 2
        bar_h = 8.0
        
        pdf.set_fill_color(220, 220, 220)
        pdf.rect(start_x, start_y, max_bar_w, bar_h, style='F')
        
        curr_x = start_x
        scale = max_bar_w / float(s_len)
        
        for c_val in cuts:
            w_mm = c_val * scale
            r, g, b = color_map[c_val]["rgb"]
            pdf.set_fill_color(r, g, b)
            pdf.rect(curr_x, start_y, w_mm, bar_h, style='DF')
            
            if w_mm > 10:
                pdf.set_text_color(255, 255, 255)
                pdf.set_font("Helvetica", "B", 7)
                pdf.set_xy(curr_x, start_y + 1.5)
                pdf.cell(w_mm, 5, f"{c_val}", align="C")
            curr_x += (c_val + kerf_width) * scale
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_y(start_y + bar_h + 4)
        pdf.ln(2)

    res = pdf.output(dest='S')
    return res.encode('latin1') if isinstance(res, str) else bytes(res)

# UI Streamlit
st.title("✂️ โปรแกรมคำนวณตัดเหล็กถอดแบบ (1D Cutting Optimization)")
st.caption("วางแผนตัดเหล็ก ประหยัดเศษเหลือ จัดกลุ่มพาเนลซ้ำ แยกสีให้ช่างดูง่าย และออกรายงาน PDF")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. สต็อกเหล็กที่มี (Stock Input)")
    kerf_width = st.number_input("ความหนาใบตัด / Kerf (มม.)", min_value=0, max_value=20, value=3)
    
    stock_df = st.data_editor(
        pd.DataFrame({"ความยาว (มม.)": [6000, 3500], "จำนวนที่มี (เส้น)": [50, 20]}),
        num_rows="dynamic",
        key="stock_editor"
    )

with col_right:
    st.subheader("2. รายการชิ้นงานที่ต้องการตัด (Cut Demands)")
    st.write("")
    st.write("")
    demand_df = st.data_editor(
        pd.DataFrame({"ความยาวที่ต้องการ (มม.)": [1500, 1200, 800], "จำนวนชิ้น": [12, 10, 8]}),
        num_rows="dynamic",
        key="demand_editor"
    )

if st.button("🚀 คำนวณแผนการตัดเหล็ก", type="primary", use_container_width=True):
    stocks = []
    for _, r in stock_df.iterrows():
        length = r.get("ความยาว (มม.)")
        qty = r.get("จำนวนที่มี (เส้น)")
        if pd.notna(length) and pd.notna(qty) and length > 0 and qty > 0:
            stocks.append((int(length), int(qty)))
            
    demands = []
    for _, r in demand_df.iterrows():
        length = r.get("ความยาวที่ต้องการ (มม.)")
        qty = r.get("จำนวนชิ้น")
        if pd.notna(length) and pd.notna(qty) and length > 0 and qty > 0:
            demands.append((int(length), int(qty)))
    
    if not stocks or not demands:
        st.warning("⚠️ กรุณากรอกข้อมูลความยาวและจำนวนให้ครบถ้วนอย่างน้อย 1 รายการ")
    else:
        with st.spinner('กำลังคำนวณหาวิธีที่ประหยัดที่สุด...'):
            results = optimize_cutting_multi_stock(stocks, kerf_width, demands)
        
        if isinstance(results, str):
            if results == "INFEASIBLE":
                st.error("❌ ไม่สามารถคำนวณได้: จำนวนเหล็กสต็อกไม่เพียงพอ หรือ มีชิ้นงานที่ยาวกว่าเหล็กสต็อก")
            else:
                st.error(f"❌ พบข้อผิดพลาด: {results}")
        else:
            pattern_counts = Counter(results)
            
            grouped_patterns = []
            stock_summary = {}
            demand_summary = Counter()

            for (s_len, cuts), count in pattern_counts.items():
                stock_summary[s_len] = stock_summary.get(s_len, 0) + count
                for c in cuts:
                    demand_summary[c] += count
                used = sum(cuts) + (len(cuts) - 1) * kerf_width
                waste = s_len - used
                grouped_patterns.append({
                    "stock_length": s_len,
                    "cuts": list(cuts),
                    "count": count,
                    "used": used,
                    "waste": waste
                })

            color_map = get_color_map(demand_summary.keys())

            st.markdown("---")
            st.header("📊 สรุปผลการตัดเหล็ก")
            
            st.subheader("1. สรุปจำนวนเหล็กเส้นที่ต้องใช้ (แยกตามความยาวสต็อก)")
            m_cols = st.columns(len(stock_summary) + 1)
            total_bars = sum(stock_summary.values())
            m_cols[0].metric("รวมเหล็กที่ใช้ทั้งหมด", f"{total_bars} เส้น")
            for idx, (s_len, qty) in enumerate(sorted(stock_summary.items()), 1):
                m_cols[idx].metric(f"เหล็กยาว {s_len:,} มม.", f"{qty} เส้น")

            st.subheader("2. สัญลักษณ์สีแยกตามขนาดความยาวชิ้นงาน")
            badge_html = ""
            for cut_len, info in sorted(color_map.items()):
                qty = demand_summary[cut_len]
                hex_color = info["hex"]
                badge_html += f'<span style="background-color:{hex_color}; color:white; padding:4px 10px; border-radius:12px; margin-right:8px; font-weight:bold; display:inline-block; margin-bottom:6px;">ตัดยาว {cut_len:,} มม. (จำนวน {qty} ชิ้น)</span> '
            st.markdown(badge_html, unsafe_allow_html=True)
            st.write("")

            st.subheader("3. ผัง Pattern การตัด (รวมรูปแบบซ้ำและคูณจำนวนเส้น)")
            for i, p in enumerate(grouped_patterns, 1):
                st.markdown(f"##### 🔹 Pattern #{i} — เหล็กยาว **{p['stock_length']:,} มม.** | <span style='color:red; font-weight:bold;'>ตัดแบบนี้ทั้งหมด {p['count']} เส้น</span>", unsafe_allow_html=True)
                st.caption(f"ความยาวที่ใช้: {p['used']:,} มม. | เศษเหลือ: {p['waste']:,} มม.")
                
                bar_html = f'<div style="display:flex; width:100%; height:36px; background-color:#e0e0e0; border-radius:6px; overflow:hidden; border:1px solid #ccc;">'
                for c_val in p['cuts']:
                    pct = (c_val / p['stock_length']) * 100
                    c_hex = color_map[c_val]["hex"]
                    bar_html += f'<div style="width:{pct}%; background-color:{c_hex}; color:white; font-weight:bold; font-size:12px; display:flex; align-items:center; justify-content:center; border-right:2px solid white;">{c_val}</div>'
                bar_html += '</div>'
                st.markdown(bar_html, unsafe_allow_html=True)
                st.write("")

            pdf_data = generate_pdf(grouped_patterns, stock_summary, demand_summary, kerf_width, color_map)
            st.download_button(
                label="📄 ดาวน์โหลดใบสั่งตัดงาน (PDF Job Sheet)",
                data=pdf_data,
                file_name="cutting_plan_jobsheet.pdf",
                mime="application/pdf",
                use_container_width=True
            )
