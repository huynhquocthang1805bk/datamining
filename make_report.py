"""Make an academic Word/Markdown report from actual saved metrics.

No experiments -> placeholders, never invented scores.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from tgcn.utility import ROOT

DATA_INFO={
 'losloop':('Los-loop','207 cảm biến, 2016 mốc thời gian, 5 phút/mốc',5),
 'sztaxi':('SZ-taxi','156 tuyến đường, 2976 mốc thời gian, 15 phút/mốc',15),
 'demo':('Demo tổng hợp','Dữ liệu tổng hợp để kiểm thử mã (không dùng chứng minh nghiên cứu)',15),
}


def fetch_rows(dataset):
    base=ROOT/'results'/dataset
    rows=[]
    for fp in sorted(base.glob('**/metrics.json')):
        obj=json.loads(fp.read_text(encoding='utf-8'))
        if obj.get('dataset') != dataset or obj.get('caution'):
            continue
        config_path=fp.parent/'config.json'
        if config_path.is_file():
            config=json.loads(config_path.read_text(encoding='utf-8'))
            par=config.get('args',{})
            if par.get('max_train_batches') or par.get('max_val_batches'):
                continue
        row={'name':fp.parent.relative_to(base).as_posix(),
             'model':obj['model'],'seed':obj.get('seed','?'),
             'MAE':obj['overall']['MAE'], 'RMSE':obj['overall']['RMSE'],
             'MAPE (%)':obj['overall']['MAPE_percent'],
             'WAPE (%)':obj['overall']['WAPE_percent'],
             'best_epoch':obj.get('best_epoch','?')}
        rows.append(row)
    return rows


def mdreport(dataset,rows):
    name,detail,interval=DATA_INFO[dataset]
    report=f'''# DỰ BÁO TỐC ĐỘ GIAO THÔNG BẰNG T-GCN VÀ T-GCN CẢI TIẾN

## 1. Giới thiệu
Dự báo tốc độ giao thông nhiều vị trí là bài toán chuỗi thời gian có phụ thuộc không gian.
T-GCN kết hợp Graph Convolutional Network (GCN) và Gated Recurrent Unit (GRU),
nhằm khai thác đồng thời kết nối giữa các cảm biến và biến thiên theo thời gian [1].
Mục tiêu đồ án: cài đặt lại mô hình, so sánh baseline GRU với T-GCN,
thử một biến thể T-GCN+ và kiểm tra từng thành phần cải tiến.

## 2. Dữ liệu và tiền xử lý
Bộ dữ liệu: **{name}** ({detail}).
Chỉ dùng dữ liệu tốc độ và ma trận kề; không tự nhận có nhãn kẹt xe, thời tiết hoặc lưu lượng.
Chia theo thời gian: 70% train, 10% validation và 20% test (không trộn ngẫu nhiên).
Dùng z-score theo cảm biến với trung bình và độ lệch chuẩn **chỉ tính từ train**.
Cửa sổ vào mặc định dài 12 bước; dự báo trực tiếp 3 bước.
Mỗi bước là {interval} phút. Mô hình có thể dùng bước lịch tuần hoàn nếu biết ngày giờ bắt đầu.

## 3. Phương pháp
**T-GCN cơ sở:** Với A là ma trận kề, đặt A_tilde = A + I (chỉ thêm self-loop nếu chưa có),
A_bar = D^(-1/2) A_tilde D^(-1/2). Mỗi cổng GRU dùng phép biến đổi
GCN(X) = A_bar X W + b. Với trạng thái H tại thời điểm t-1:

z_t = sigmoid(GCN([x_t, H_(t-1)]));
r_t = sigmoid(GCN([x_t, H_(t-1)]));
H~_t = tanh(GCN([x_t, r_t * H_(t-1)]));
H_t = z_t * H_(t-1) + (1-z_t) * H~_t.

Một lớp tuyến tính biến trạng thái cuối thành dự báo đa bước tại mỗi nút.
Baseline GRU chỉ học diễn biến chuỗi tại từng nút mà không đưa ma trận kề vào mô hình.

**T-GCN+ (giả thuyết cải tiến):**

1. Học đồ thị mềm A_learn = softmax(ReLU(E E^T)), rồi phối hợp với A_bar qua alpha học được.
2. Bổ sung sin/cos của giờ trong ngày và thứ trong tuần.
3. Head dự báo phần thay đổi, cộng với tốc độ gần nhất (residual forecasting).
4. Dùng dropout và early stopping bằng validation MSE; thử Huber (Smooth L1) riêng nếu cần.

Cần thực nghiệm ablation để quy kết tác dụng từng thành phần; chưa thể khẳng định cải tiến tốt hơn.

## 4. Thiết kế thí nghiệm
Huấn luyện tất cả mô hình trên cùng bộ dữ liệu, cùng horizon, cùng train/val/test và seed.
Chọn checkpoint theo MSE nhỏ nhất trên validation; chỉ dùng test để báo cáo sau cùng.
Đánh giá MAE, RMSE, WAPE, MAPE (với |giá trị thật| >= 5 đơn vị; phải ghi tỷ lệ bao phủ).
Ưu tiên tối thiểu ba seed và báo cáo trung bình ± độ lệch chuẩn.
Tốc độ dùng đơn vị do nguồn dữ liệu cung cấp, cần đối chiếu metadata khi công bố.

## 5. Kết quả thực nghiệm
'''
    if rows:
        report+='\n| Run | Seed | MAE | RMSE | MAPE (%) | WAPE (%) |\n|---|---:|---:|---:|---:|---:|\n'
        for r in rows:
            report+=f"| {r['name']} | {r['seed']} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {('%.2f'%r['MAPE (%)']) if r['MAPE (%)'] is not None else 'N/A'} | {r['WAPE (%)']:.2f} |\n"
        report+='\nCác chỉ số được đọc từ file metrics.json sau khi huấn luyện và đánh giá.\n'
    else:
        report+='\n*Chưa có kết quả huấn luyện đầy đủ cho dữ liệu này. Bảng MAE/RMSE/MAPE sẽ tự cập nhật sau khi chạy `python make_report.py --dataset '+dataset+'`.*\n'
    report+='''
## 6. Thảo luận và giới hạn
- Phân tích lỗi theo chân trời dự báo và từng vị trí; trình bày ví dụ biểu đồ chuỗi thời gian.
- So sánh GRU với T-GCN để đánh giá vai trò thông tin đồ thị.
- So sánh T-GCN với T-GCN+ và từng phép bỏ thành phần để đánh giá đóng góp thực sự.
- Không suy diễn rằng MAE/RMSE nhỏ đồng nghĩa phát hiện tắc nghẽn chính xác: đây là bài toán **dự báo tốc độ**, chưa có phân loại tắc nghẽn.
- Mô hình gốc ở đây là cài đặt độc lập, khác thiết lập gốc về chuẩn hóa, chia tập, thời gian dự báo và framework; vì vậy không trực tiếp so chỉ số với bài báo.
- Đặc trưng giờ/ngày được xây dựng từ mốc thời gian và giả định lấy mẫu liên tục; nếu có thiếu thời điểm phải chỉnh lại.
- Graph được đối xứng hóa; nghiên cứu đồ thị có hướng cần kiểm tra định nghĩa cạnh thực tế.

## 7. Kết luận
Đánh giá đóng góp chỉ dựa trên kết quả thực nghiệm của dữ liệu thật, có kiểm soát seed và ablation.
Nếu T-GCN+ không cải thiện hoặc độ chênh lệch thấp hơn độ biến động giữa seed, cần báo cáo trung thực.

## Tài liệu tham khảo
[1] Zhao, L. et al. (2019). T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction.
IEEE Transactions on Intelligent Transportation Systems. https://arxiv.org/abs/1811.05320

[2] Mã nguồn và dữ liệu của nhóm tác giả: https://github.com/lehaifeng/T-GCN
'''
    return report


def style_doc(doc):
    sec=doc.sections[0];sec.top_margin=Cm(2.5);sec.bottom_margin=Cm(2.5)
    sec.left_margin=Cm(3.0);sec.right_margin=Cm(2.4)
    normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(12)
    normal.paragraph_format.space_after=Pt(5);normal.paragraph_format.line_spacing=1.23
    for level,size in [('Title',17),('Heading 1',14),('Heading 2',12)]:
        s=doc.styles[level];s.font.name='Times New Roman';s.font.size=Pt(size)
        s.font.bold=True;s.font.color.rgb=RGBColor(26,47,70)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=footer.add_run('Dự báo tốc độ giao thông — T-GCN');r.font.name='Times New Roman';r.font.size=Pt(9)


def heading(doc,title): doc.add_heading(title,level=1)
def para(doc,text): doc.add_paragraph(text)


def make_word(dataset,rows,output):
    name,detail,interval=DATA_INFO[dataset]
    d=Document();style_doc(d)
    cover=d.add_paragraph();cover.alignment=WD_ALIGN_PARAGRAPH.CENTER
    cover.paragraph_format.space_before=Pt(65)
    r=cover.add_run('BÁO CÁO ĐỒ ÁN\nDỰ BÁO TỐC ĐỘ GIAO THÔNG\nBẰNG T-GCN VÀ BIẾN THỂ CẢI TIẾN')
    r.bold=True;r.font.name='Times New Roman';r.font.size=Pt(18)
    c=d.add_paragraph('Mô hình: GRU – T-GCN – T-GCN+');c.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for x in ['Họ và tên: ...................................................',
              'Mã số sinh viên: ...........................................',
              'Học phần / Lớp: .............................................',
              'Giảng viên hướng dẫn: ...................................']:
        p=d.add_paragraph(x);p.paragraph_format.space_before=Pt(12)
    c=d.add_paragraph('Tháng 11 năm 2026');c.alignment=WD_ALIGN_PARAGRAPH.CENTER
    d.add_page_break()
    heading(d,'1. Giới thiệu và mục tiêu')
    para(d,'Dự báo vận tốc giao thông tại nhiều cảm biến là bài toán chuỗi thời gian đa biến có phụ thuộc theo không gian và thời gian. T-GCN kết hợp mạng tích chập đồ thị (GCN) với mạng GRU để học đồng thời hai dạng phụ thuộc này [1].')
    para(d,'Mục tiêu: (i) tái triển khai độc lập mô hình T-GCN bằng PyTorch; (ii) đối chiếu baseline GRU; (iii) xây dựng T-GCN+ với các cơ chế mở rộng; (iv) đo sai số dự báo và thực hiện ablation để đánh giá đóng góp của các cải tiến.')
    heading(d,'2. Dữ liệu và xử lý')
    para(d,f'Bộ dữ liệu: {name}. Đặc điểm: {detail}. Đầu vào gồm ma trận tốc độ theo thời gian và ma trận kề của mạng cảm biến. Bộ dữ liệu không mặc nhiên cung cấp nhãn phân loại tắc nghẽn.')
    para(d,'Chia dữ liệu theo đúng thứ tự thời gian: 70% train, 10% validation, 20% test. Phép chuẩn hóa z-score theo mỗi cảm biến chỉ dùng trung bình và độ lệch chuẩn của tập train. Mỗi sample dùng 12 bước quá khứ để dự báo 3 bước tiếp theo, tương ứng '+f'{interval}, {2*interval}, {3*interval} phút.')
    para(d,'Các mẫu validation và test có thể dùng quá khứ ngay trước ranh giới tập để tạo cửa sổ đầu vào, nhưng nhãn dự báo phải nằm hoàn toàn bên trong tập tương ứng; không trộn dữ liệu theo thời gian.')
    heading(d,'3. Kiến trúc mô hình')
    d.add_heading('3.1. T-GCN',level=2)
    para(d,'GCN khai thác liên kết mạng đường bằng phép lan truyền trên đồ thị: Ā = D^(−1/2) Ã D^(−1/2), với Ã là ma trận kề có self-loop. Tại mỗi thời điểm, đầu vào x(t) và trạng thái ẩn h(t−1) được đưa vào các cổng GRU sau biến đổi GCN.')
    para(d,'z(t) = σ(GCN([x(t), h(t−1)])) ; r(t) = σ(GCN([x(t), h(t−1)]))')
    para(d,'h̃(t) = tanh(GCN([x(t), r(t) ⊙ h(t−1)]))')
    para(d,'h(t) = z(t) ⊙ h(t−1) + (1 − z(t)) ⊙ h̃(t)')
    para(d,'Sau bước cuối, một tầng tuyến tính tạo dự báo đồng thời H bước cho tất cả cảm biến. GRU baseline được dùng độc lập cho mỗi cảm biến và không tận dụng ma trận kề.')
    d.add_heading('3.2. T-GCN+ và giả thuyết cải tiến',level=2)
    para(d,'Biến thể đề xuất gồm: (1) đồ thị mềm học từ embedding của các cảm biến và pha trộn với đồ thị vật lý; (2) đặc trưng sin/cos của giờ trong ngày và ngày trong tuần; (3) dự báo phần dư so với tốc độ quan sát gần nhất; (4) dropout và early stopping theo validation MSE. Thử Huber như một ablation riêng; để so sánh chính dùng MSE.')
    para(d,'Đồ thị thích nghi: A_learn = softmax(ReLU(E Eᵀ)); A_mix = (1 − α) A_physical + α A_learn, α học từ dữ liệu. Đây là mở rộng nghiên cứu của dự án, không phải thành phần nguyên bản của T-GCN [1].')
    para(d,'Các cơ chế trên chỉ là giả thuyết cải tiến. Cần kết quả test trên cùng bộ dữ liệu và thí nghiệm loại từng thành phần để xác nhận hiệu quả.')
    heading(d,'4. Thiết lập thực nghiệm và chỉ số')
    para(d,'Huấn luyện với AdamW, learning rate 0,001, batch size 32, early stopping theo validation MSE. Chạy cùng train/validation/test, cùng horizon và nhiều seed để đánh giá độ ổn định. Điểm trên test chỉ được tính bằng checkpoint chọn trên validation.')
    para(d,'MAE = mean(|ŷ − y|); RMSE = sqrt(mean((ŷ − y)²)); WAPE = 100 × sum(|ŷ − y|) / sum(|y|). MAPE chỉ tính với |y| ≥ 5 đơn vị tốc độ để tránh chia cho số gần 0; luôn ghi rõ mức bao phủ.')
    heading(d,'5. Kết quả thực nghiệm')
    table=d.add_table(rows=1, cols=6);table.style='Table Grid';table.alignment=WD_TABLE_ALIGNMENT.CENTER
    titles=['Mô hình / lần chạy','Seed','MAE','RMSE','MAPE (%)','WAPE (%)']
    for i,t in enumerate(titles): table.rows[0].cells[i].text=t
    if rows:
        for row in rows:
            c=table.add_row().cells
            fmt=lambda v: '—' if v is None else f'{v:.3f}'
            for i,val in enumerate([row['name'],str(row['seed']),fmt(row['MAE']),fmt(row['RMSE']),
                                     fmt(row['MAPE (%)']),fmt(row['WAPE (%)'])]): c[i].text=val
    else:
        cells=table.add_row().cells
        cells[0].text='Chưa chạy dữ liệu thật'
        for i in range(1,6):cells[i].text='—'
    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for pp in cell.paragraphs:
                for r in pp.runs:r.font.name='Times New Roman';r.font.size=Pt(9)
    if rows:
        para(d,'Các số đo lấy từ metrics.json do chương trình tạo từ các dự báo trên tập test. Chỉ số có thể khác tác giả vì split, chuẩn hóa và cách cài đặt không trùng khớp.')
        best=min(rows,key=lambda r:r['RMSE'])
        para(d,'Trong các run đã hoàn thành, '+best['name']+f" có RMSE thấp nhất: {best['RMSE']:.3f}. Không thể kết luận ưu thế chung nếu chưa xét độ biến động theo seed và ablation.")
        rpath=ROOT/'results'/dataset/best['name']/'forecast_sensor.png'
        if rpath.is_file():
            d.add_picture(str(rpath),width=Inches(5.9))
            p=d.add_paragraph(f'Hình 1. Tốc độ thực và dự báo (sensor mẫu, {best["name"]}).')
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    else:
        para(d,'Chưa có số đo thực nghiệm hoàn chỉnh. Sau khi chạy huấn luyện trên dữ liệu thật, dùng lệnh python make_report.py --dataset '+dataset+' để tự động cập nhật bảng và chèn hình.')
    heading(d,'6. Thảo luận, hạn chế và kiểm chứng')
    para(d,'Cần phân tích sai số theo các thời điểm dự báo và theo từng cảm biến; đối chiếu GRU với T-GCN để đánh giá đóng góp của đồ thị; sau đó so T-GCN với T-GCN+ trong các thí nghiệm ablation. Không khẳng định mô hình phát hiện tắc nghẽn vì mục tiêu thí nghiệm là dự báo tốc độ liên tục.')
    para(d,'Các giới hạn: không bao gồm thời tiết và lưu lượng; mốc thời gian được dựng từ ngày bắt đầu và giả định lấy mẫu đều; ma trận kề đã được đối xứng hóa; mô hình được cài đặt độc lập bằng PyTorch với giao thức dữ liệu khác mã TensorFlow gốc. Không so sánh trực tiếp chỉ số với bài báo khi điều kiện khác nhau.')
    heading(d,'7. Kết luận')
    para(d,'Kết luận cuối cùng chỉ nên viết sau khi có MAE/RMSE trên dữ liệu thật, cùng phép so sánh nhiều seed và ablation. Nếu biến thể T-GCN+ không vượt baseline, cần phản ánh đúng và phân tích nguyên nhân thay vì mặc định cải tiến thành công.')
    heading(d,'Tài liệu tham khảo')
    para(d,'[1] Zhao, L. và cộng sự (2019). T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction. IEEE Transactions on Intelligent Transportation Systems. https://arxiv.org/abs/1811.05320')
    para(d,'[2] T-GCN GitHub Repository. https://github.com/lehaifeng/T-GCN')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);d.save(output)
    print('Word report:',output)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATA_INFO,default='losloop')
    args=p.parse_args();rows=fetch_rows(args.dataset)
    report_dir=ROOT/'reports';report_dir.mkdir(exist_ok=True)
    md=report_dir/f'BAO_CAO_{args.dataset.upper()}.md'
    md.write_text(mdreport(args.dataset,rows),encoding='utf-8')
    make_word(args.dataset,rows,report_dir/f'BAO_CAO_{args.dataset.upper()}.docx')
    print(f'Markdown report: {md}, valid full experiments: {len(rows)}')

if __name__=='__main__':main()
