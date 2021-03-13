# Multi-label Classification

### Hướng dẫn chạy code
#### Preprocess data
Trước khi chạy model, cần tokenize data trước. Mục đích để dễ debug và giảm thời gian chạy code, nếu mỗi lần các bạn chạy model lại tokenize 1 lần thì sẽ rất tốn thời gian, đặc biệt có những task thời gian preprocess và tokenization lên tới hàng giờ đồng hồ.
python script/prepare_data.py

#### Train model
python train.py +experiment=exp_intent.yaml

exp_intent.yaml nằm trong folder configs (folder này chứa toàn bộ config về data, model, traner, etc của project)
Điều đặc biệt và 
