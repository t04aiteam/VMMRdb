Đây là tổng quan về bộ dữ liệu VMMR được giới thiệu trong bài báo "[A Large and Diverse Dataset for Improved Vehicle Make and Model Recognition](https://github.com/faezetta/VMMRdb/blob/master/meta/VMMR_TSWC.pdf)".

<p align="center"><img align="center" src="https://github.com/faezetta/VMMRdb/blob/master/meta/vmmrMultiplicity.png" alt="VMMRdb examples of multiplicity" width="400px">
<img align="center" src="https://github.com/faezetta/VMMRdb/blob/master/meta/vmmrAmbiguity.png" alt="VMMRdb examples af ambiguity" width="400px"></p>

## Tổng quan
Mặc dù có nhiều nghiên cứu và nhu cầu thực tiễn, bài toán nhận diện hãng xe và mẫu xe (make/model) vẫn chưa nhận được nhiều sự quan tâm trong cộng đồng thị giác máy tính. Chúng tôi cho rằng việc thiếu các bộ dữ liệu chất lượng cao đã hạn chế đáng kể khả năng khai phá của cộng đồng trong lĩnh vực này. Vì vậy, chúng tôi đã thu thập và tổ chức một cơ sở dữ liệu hình ảnh quy mô lớn và toàn diện có tên VMMRdb, trong đó mỗi hình ảnh được gán nhãn hãng xe, mẫu xe và năm sản xuất tương ứng.

## Mô tả
Bộ dữ liệu Nhận diện Hãng xe và Mẫu xe (VMMRdb) có quy mô và độ đa dạng lớn, gồm 9.170 lớp (class) với tổng cộng 291.752 hình ảnh, bao phủ các mẫu xe được sản xuất từ năm 1950 đến 2016. Bộ dữ liệu VMMRdb chứa hình ảnh được chụp bởi nhiều người dùng khác nhau, nhiều thiết bị chụp ảnh khác nhau, và nhiều góc nhìn khác nhau, đảm bảo sự đa dạng cần thiết để phản ánh các tình huống có thể gặp trong thực tế. Các xe trong ảnh không được căn chỉnh chuẩn, và một số hình ảnh có chứa nền/hậu cảnh không liên quan. Dữ liệu bao phủ các phương tiện từ 712 khu vực, thuộc toàn bộ 412 tiểu vùng tương ứng với các khu đô thị (metro area) tại Mỹ. Bộ dữ liệu của chúng tôi có thể được dùng làm nền tảng (baseline) để huấn luyện một mô hình mạnh mẽ cho nhiều tình huống thực tế trong giám sát giao thông.
<p align="center"><img align="center" src="https://github.com/faezetta/VMMRdb/blob/master/meta/dbHeatmap.png" alt="VMMRdb data distribution" width="500px"></p>
<p align="center"><sub>Phân bố hình ảnh theo từng lớp trong bộ dữ liệu. Mỗi hình tròn tương ứng với một lớp, kích thước hình tròn thể hiện số lượng hình ảnh trong lớp đó. Các lớp có gắn nhãn là những lớp có hơn 100 hình ảnh.</sub></p>

## Tải xuống
VMMRdb có thể được tải xuống [tại đây](https://www.dropbox.com/s/uwa7c5uz7cac7cw/VMMRdb.zip?dl=0).

Mỗi hình ảnh được gán nhãn hãng xe, mẫu xe và năm sản xuất tương ứng.

Một số mô hình được nhắc đến trong bài báo của chúng tôi trên VMMRdb-3036: [Resnet-50](https://www.dropbox.com/s/vt4svttnpshyovv/resnet-50.t7?dl=0), [VGG](https://www.dropbox.com/s/l38ik039s5rm0w5/vgg.t7?dl=0)

## Trích dẫn
Nếu bạn sử dụng bộ dữ liệu này, vui lòng trích dẫn bài báo sau:
```
A Large and Diverse Dataset for Improved Vehicle Make and Model Recognition
F. Tafazzoli, K. Nishiyama and H. Frigui
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops 2017.
```
