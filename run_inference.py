# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

#--
import time
from googletrans import Translator
import cv2
from PIL import Image, ImageFilter
import numpy as np
import threading
#--

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")

tf.logging.set_verbosity(tf.logging.INFO)

text = ['hoge', 'fuga', 'piyo']

def main(_):

  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)


  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.

    generator = caption_generator.CaptionGenerator(model, vocab)
    translator = Translator()

    #---
    DEVICE_ID = 0
    capture = cv2.VideoCapture(DEVICE_ID)
    if capture.isOpened() is False:
        raise ("IO Error")
    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)


    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_color = (0, 255, 0)
    #---

    while True:
      ret, img = capture.read()

      if (threading.activeCount() == 1):
          th = CaptionThread(sess, generator, translator, vocab, img)
          th.start()

      for i, txt in enumerate(text):
          cv2.putText(img, txt, (10, 30*(i+1)), font, font_size, font_color) # 日本語のフォントが無い？

      cv2.imshow("Capture", img)

      key = cv2.waitKey(1)
      if key == 27: # ESC key
          break

      if ret == False:
          continue

    cv2.destroyAllWindows()
    cap.release()


class CaptionThread(threading.Thread):
    def __init__(self, sess, generator, translator, vocab, image):
        super(CaptionThread, self).__init__()
        self._sess = sess
        self._generator = generator
        self._translator = translator
        self._vocab = vocab
        self._image = image

    def run(self):
        starttime = time.time()
        # with tf.gfile.GFile('./tmp.png', "rb") as f:
        #     image = f.read()


        image_tensor = tf.convert_to_tensor(np.uint8(self._image[:, :, ::-1].copy()))
        with tf.Session() as sess: # sessionをネストして大丈夫なのか?
            encoded = tf.image.encode_jpeg(image_tensor)
            encoded_data = sess.run(encoded)
            # # 確認用
            # decoded = tf.image.decode_jpeg(encoded)
            # decoded_data = sess.run(decoded)
            # print(decoded_data)
            # pil_img_f = Image.fromarray(np.uint8(decoded_data))
            # pil_img_f.save('./temp.jpeg')

        #print("encoding image time :", time.time() - starttime)

        captions = self._generator.beam_search(self._sess, encoded_data)

        print("Captions for image:")
        for i, caption in enumerate(captions):
            # Ignore begin and end words.
            sentence = [self._vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            sentence_transed = self._translator.translate(sentence, dest='ja').text
            print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            print("    transed: %s" % sentence_transed)

            global text
            text[i] = ("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))


        elapsedtime = time.time() - starttime
        print("elapsed time:", elapsedtime, "[sec]")

if __name__ == "__main__":
  tf.app.run()
