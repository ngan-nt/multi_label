# Multi-label Classification

Hi, mình sẽ hướng dẫn các bạn cách làm bài Multi-Label nhé.

### Requirements
Trong project này, mình sẽ sử dụng PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).
Lý do mình sử dụng framework này là bởi vì nó hỗ trợ rất nhiều trong việc chạy model dùng CPU, GPU, multi-GPU, ngoài ra mình thấy cách viết code của PyTorch Lightning sẽ giúp mình thiết kế code sạch, đẹp, một khung code có thể sử dụng cho nhiều project, đỡ tốn thời gian và công sưc.
Ngoài ra, code của mình được dựa trên template [này](https://github.com/hobogalaxy/lightning-hydra-template). Các bạn đọc thêm repo trên để hiểu hơn về cấu trúc code nhé.

Các package version mà mình sử dụng:
- Python 3.7
- Pytorch 1.7
- Pytorch-lightning 1.2.2
- Hydra 1.1

### Hướng dẫn chạy code
#### Preprocess data
Trước khi chạy model, cần tokenize data trước. Mục đích để dễ debug và giảm thời gian chạy code, nếu mỗi lần các bạn chạy model lại tokenize 1 lần thì sẽ rất tốn thời gian, đặc biệt có những task thời gian preprocess và tokenization lên tới hàng giờ đồng hồ.
```bash
python script/prepare_data.py
```
* Lưu ý: ở đây mình tự viết hàm tokenizer riêng phù hợp với pretrained model của mình, các bạn có thể sử dụng PhoBert hay bất kì pretrained model nào.


#### Train model
```bash
python train.py +experiment=exp_intent.yaml
```
exp_intent.yaml nằm trong folder configs (folder này chứa toàn bộ config về data, model, traner, etc của project)
Chi tiết các bạn xem thêm ở repo có template mình đính kèm phía trên nhé.

#### Inference 
Output sẽ lưu ở data/processed_data.
```bash
python inference.py
```
