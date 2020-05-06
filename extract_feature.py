import modeling
import tokenization
from graph import optimize_graph
import args
from queue import Queue
from threading import Thread
import tensorflow as tf
import os
import jieba
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


class BertVector:

    def __init__(self, batch_size=32):
        """
        init BertVector
        :param batch_size:     Depending on your memory default is 32
        """
        self.max_seq_length = args.max_seq_len
        self.layer_indexes = args.layer_indexes
        self.gpu_memory_fraction = 1
        if os.path.exists(args.graph_file):
            self.graph_path = args.graph_file
        else:
            self.graph_path = optimize_graph()

        self.tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
        self.batch_size = batch_size
        self.estimator = self.get_estimator()
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.predict_thread = Thread(target=self.predict_from_queue, daemon=True)
        self.predict_thread.start()

    def get_estimator(self):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec

        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            input_names = ['input_ids', 'input_mask', 'input_type_ids']
            # batch = tf.data.Dataset.range(10).make_one_shot_iterator().get_next()
            # try:
            #     while True:
            #         print(sess.run(y))
            # except tf.errors.OutOfRangeError:
            #     pass
            output = tf.import_graph_def(graph_def,
                                         input_map={k + ':0': features[k] for k in input_names},
                                         return_elements=['final_encodes:0', 'final_encodes2:0'])

            return EstimatorSpec(mode=mode, predictions={
                'encodes': output[0], 'encodes2': output[1]
            })

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config),
                         params={'batch_size': self.batch_size}, model_dir='../tmp')

    def predict_from_queue(self):
        prediction = self.estimator.predict(input_fn=self.queue_predict_input_fn, yield_single_examples=False)
        for i in prediction:
            a = i['encodes2'][0, 5:, :]
            self.output_queue.put(i)

    def encode(self, sentence):
        self.input_queue.put(sentence)
        prediction = self.output_queue.get()['encodes2']
        return prediction

    def queue_predict_input_fn(self):

        return (tf.data.Dataset.from_generator(
            self.generate_from_queue,
            output_types={'unique_ids': tf.int32,
                          'input_ids': tf.int32,
                          'input_mask': tf.int32,
                          'input_type_ids': tf.int32},
            output_shapes={
                'unique_ids': (None,),
                'input_ids': (None, self.max_seq_length),
                'input_mask': (None, self.max_seq_length),
                'input_type_ids': (None, self.max_seq_length)}).prefetch(10))

    def generate_from_queue(self):
        while True:
            features = list(self.convert_examples_to_features(seq_length=self.max_seq_length, tokenizer=self.tokenizer))
            yield {
                'unique_ids': [f.unique_id for f in features],
                'input_ids': [f.input_ids for f in features],
                'input_mask': [f.input_mask for f in features],
                'input_type_ids': [f.input_type_ids for f in features]
            }

    def convert_examples_to_features(self, seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        input_masks = []
        examples = self._to_example(self.input_queue.get())
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            # if the sentences's length is more than seq_length, only use sentence's left part
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            # Where "input_ids" are tokens's index in vocabulary
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            input_masks.append(input_mask)
            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            if ex_index < 5:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (example.unique_id))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

            yield InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids)


    @staticmethod
    def _to_example(sentences):
        import re
        """
        sentences to InputExample
        :param sentences: list of strings
        :return: list of InputExample
        """
        unique_id = 0
        for ss in sentences:
            line = tokenization.convert_to_unicode(ss)
            if not line:
                continue
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
            unique_id += 1


def get_word_embedding(sentence, sentence_embedding):
    word_embedding_list = {}
    word_list = jieba.lcut(sentence)
    print(word_list)
    char_index = 0
    # 去掉开头的[CLS]
    sentence_embedding = np.squeeze(sentence_embedding[:, 1:, :])
    for word in word_list:
        word_length = len(word)
        word_embedding = np.zeros(sentence_embedding.shape[1])
        for embedding_index in range(char_index, char_index+word_length):
            word_embedding += np.squeeze(sentence_embedding[embedding_index, :])
        word_embedding /= word_length
        word_embedding_list[word] = word_embedding
        char_index += word_length
    return word_embedding_list


def main():
    bert = BertVector()
    # question = input('question: ')
    sentence_list = ["小米手机用起来速度还可以哦！", "我的苹果手机现在是越来越卡了。"]
    # sentence_list = ["小米含有大量的维生素E，是大米的4.8倍；其蛋白质优于大米、小麦。除了健胃消食，常吃小米，对女性的益处大，有滋阴养颜的作用。", "据凤凰科技报道，《福布斯》刊文称，研究人员在对小米手机加载的一款浏览器进行研究时发现，浏览器会收集并分享用户隐私信息。据悉，浏览器跟踪用户几乎全部上网行为，其中包括曾经访问的网站、在谷歌网站上输入的搜索关键字、手机信息流中的所有内容。即使在“无痕”模式下，浏览器也会跟踪用户隐私信息。报道称，小米手机收集的所有信息，会发送到它在俄罗斯、新加坡设立的服务器中。"]
    # sentence_list = ["三星是一个大的手机品牌。", "小米手机用起来速度还可以哦！"]
    # sentence_list = ["玉米很容易煮的！", "这个季节产的小米不好吃。"]
    sentences_embedding = [bert.encode([x]) for x in sentence_list]
    sentence_list = [x[:bert.max_seq_length-2] if len(x) > bert.max_seq_length-2 else x for x in sentence_list]
    # for sentence, sentence_embedding in sentence_embedding_dict:
    v_1 = get_word_embedding(sentence_list[0], sentences_embedding[0])["小米"]
    v_2 = get_word_embedding(sentence_list[1], sentences_embedding[1])["苹果"]
    # 余弦相似度，值越大，越相似
    print("cosine similarity: ", np.dot(v_1, v_2)/(np.linalg.norm(v_1)*(np.linalg.norm(v_2))))
    # v = bert.encode(["小米用起来还可以哦！", "我的苹果手机现在是越来越卡了。"])
    # print(str(v))


if __name__ == "__main__":
    main()
    # bert = BertVector()
    #
    # while True:
    #     question = input('question: ')
    #     v = bert.encode([question])
        # print(str(v))
