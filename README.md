# im2txt_for_ITfes
「IT津梁まつり2018」に出展した技術デモです。
webカメラから取得した画像に対して連続的にキャプションを生成して画像中に表示します。
（たとえば「積み上がった本の上に乗ったリンゴ」）

## run_inference.py

https://github.com/tensorflow/models/tree/master/research/im2txt
の`run_inference.py`と差し替える．

* googleTranseを用いて英->日翻訳
* opencvのVideoCaptureを用いてwebカメラで取得した画像をiterativeにキャプション生成する

使い方
~~~ 
% pip install googletrans
% bazel build -c opt im2txt/run_inference
% bazel-bin/im2txt/run_inference --checkpoint_path=im2txt/Pretrained-Show-and-Tell-model/model2.ckpt-2000000 --vocab_file=im2txt/Pretrained-Show-and-Tell-model/word_counts.txt
~~~
