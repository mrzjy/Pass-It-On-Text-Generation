#! -*- coding: utf-8 -*-

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.snippets import truncate_sequences
from bert4keras.snippets import AutoRegressiveDecoder
import argparse
import jieba
import os

from modeling import RoformerLatent

jieba.initialize()

# 基本信息
latent_size = 20
maxlen = 64
batch_size = 128
steps_per_epoch = 2500
epochs = 10

# bert配置 (download from chinese_roformer-sim-char-ft_L-12_H-768_A-12)
config_path = 'data/bert_config.json'
dict_path = 'data/vocab.txt'
checkpoint_path = None

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def read(filename):
    while True:
        with open(filename) as f:
            for l in f:
                splits = l.strip().split("\t")
                if len(splits) == 2:
                    text, synonym = splits
                    yield text[:maxlen//2], synonym[:maxlen//2]


def corpus(file):
    """读取语料
    """
    d = read(file)
    while True:
        text, synonym = next(d)
        yield text, synonym


def masked_encode(text):
    """wwm随机mask
    """
    words = jieba.lcut(text)
    rands = np.random.random(len(words))
    source, target = [tokenizer._token_start_id], [0]
    for r, w in zip(rands, words):
        ids = tokenizer.encode(w)[0][1:-1]
        if r < 0.15 * 0.8:
            source.extend([tokenizer._token_mask_id] * len(ids))
            target.extend(ids)
        elif r < 0.15 * 0.9:
            source.extend(ids)
            target.extend(ids)
        elif r < 0.15:
            source.extend(
                np.random.choice(tokenizer._vocab_size - 1, size=len(ids)) + 1
            )
            target.extend(ids)
        else:
            source.extend(ids)
            target.extend([0] * len(ids))
    source = source[:maxlen - 1] + [tokenizer._token_end_id]
    target = target[:maxlen - 1] + [0]
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.some_samples = []
        self.special_mask_id = 1

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (text, synonym) in self.sample(random):
            for i in range(2):
                if np.random.random() < 0.5:
                    text_ids = masked_encode(text)[0]
                else:
                    text_ids = tokenizer.encode(text)[0]

                # add special token between CLS and text
                text_ids = text_ids[:1] + [self.special_mask_id] + text_ids[1:]

                synonym_ids = tokenizer.encode(synonym)[0][1:]
                truncate_sequences(maxlen * 2, -2, text_ids, synonym_ids)
                token_ids = text_ids + synonym_ids
                segment_ids = [0] * len(text_ids) + [1] * len(synonym_ids)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                self.some_samples.append(synonym)
                if len(self.some_samples) > 1000:
                    self.some_samples.pop(0)
                text, synonym = synonym, text
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class TotalLoss(Loss):
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。
    """
    def compute_loss(self, inputs, mask=None):
        loss1 = self.compute_loss_of_seq2seq(inputs, mask)
        loss2 = self.compute_loss_of_bow(inputs)
        loss3 = self.compute_loss_of_similarity(inputs)
        self.add_metric(loss1, name='seq2seq_loss')
        self.add_metric(loss2, name='bow_loss')
        self.add_metric(loss3, name='similarity_loss')
        return loss1 + loss2 + loss3

    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, y_cls, y_pred, y_hz = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_loss_of_bow(self, inputs, coefficient=0.001):
        y_true, y_mask, y_cls, y_pred, y_hz = inputs
        one_hot_labels = K.one_hot(K.cast(y_true, dtype="int32"), num_classes=K.int_shape(y_pred)[-1])
        multiclass_labels = K.sum(one_hot_labels, axis=1)  # [batch_size, vocab_size]
        loss = K.binary_crossentropy(multiclass_labels, y_hz, from_logits=True)
        loss = K.sum(loss) / K.cast(K.shape(y_mask)[0], "float32")
        return coefficient * loss

    def compute_loss_of_similarity(self, inputs, mask=None):
        y_true, y_mask, y_cls, y_pred, y_hz = inputs
        y_true = self.get_labels_of_similarity(y_cls)  # 构建标签
        y_cls = K.l2_normalize(y_cls, axis=1)  # 句向量归一化
        similarities = K.dot(y_cls, K.transpose(y_cls))  # 相似度矩阵
        similarities = similarities - K.eye(K.shape(y_cls)[0]) * 1e12  # 排除对角线
        similarities = similarities * 20  # scale
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = K.equal(idxs_1, idxs_2)
        labels = K.cast(labels, K.floatx())
        return labels


# 建立加载模型
roformer = build_transformer_model(
    config_path,
    checkpoint_path,
    model=RoformerLatent,
    application='custom',
    with_pool='linear',
    with_mlm='linear',
    dropout_rate=0.2,
    ignore_invalid_weights=True,
    latent_size=latent_size
)
roformer.is_infer = False
encoder = keras.models.Model(roformer.inputs, roformer.outputs[0])
seq2seq = keras.models.Model(roformer.inputs, roformer.outputs[1])

outputs = TotalLoss([2, 3, 4])(roformer.inputs + roformer.outputs)
model = keras.models.Model(roformer.inputs, outputs)

AdamW = extend_with_weight_decay(Adam, 'AdamW')
optimizer = AdamW(learning_rate=1e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)
model.summary()


class Generator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(seq2seq).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n, topp=topp)
        return [tokenizer.decode(ids) for ids in output_ids]

    def random_sample(self, inputs, n, topk=None, topp=None, states=None, temperature=1, min_ends=1):
        """随机采样latent_size个结果
        返回：n个解码序列组成的list。
        """
        token_ids, segment_ids = inputs
        batch_token_ids = []
        batch_segment_ids = []
        for latent_id in range(2, 2 + latent_size):
            for _ in range(n):
                batch_token_ids.append(
                    token_ids[:1] + [latent_id] + token_ids[1:])
                batch_segment_ids.append(
                    segment_ids[:1] + [segment_ids[1]] + segment_ids[1:])

        inputs = [batch_token_ids, batch_segment_ids]
        inputs = [np.array(i) for i in inputs]
        output_ids = np.repeat(self.first_output_ids, n * latent_size, axis=0)
        results = []
        for step in range(self.maxlen):
            probas, states = self.predict(
                inputs, output_ids, states, temperature, 'probas'
            )  # 计算当前概率
            probas /= probas.sum(axis=1, keepdims=True)  # 确保归一化
            if topk is not None:
                k_indices = probas.argpartition(-topk, axis=1)[:, -topk:]  # 仅保留topk
                probas = np.take_along_axis(probas, k_indices, axis=1)  # topk概率
                probas /= probas.sum(axis=1, keepdims=True)  # 重新归一化
            if topp is not None:
                p_indices = probas.argsort(axis=1)[:, ::-1]  # 从高到低排序
                probas = np.take_along_axis(probas, p_indices, axis=1)  # 排序概率
                cumsum_probas = np.cumsum(probas, axis=1)  # 累积概率
                flag = np.roll(cumsum_probas >= topp, 1, axis=1)  # 标记超过topp的部分
                flag[:, 0] = False  # 结合上面的np.roll，实现平移一位的效果
                probas[flag] = 0  # 后面的全部置零
                probas /= probas.sum(axis=1, keepdims=True)  # 重新归一化
            sample_func = lambda p: np.random.choice(len(p), p=p)  # 按概率采样函数
            sample_ids = np.apply_along_axis(sample_func, 1, probas)  # 执行采样
            sample_ids = sample_ids.reshape((-1, 1))  # 对齐形状
            if topp is not None:
                sample_ids = np.take_along_axis(
                    p_indices, sample_ids, axis=1
                )  # 对齐原id
            if topk is not None:
                sample_ids = np.take_along_axis(
                    k_indices, sample_ids, axis=1
                )  # 对齐原id
            output_ids = np.concatenate([output_ids, sample_ids], 1)  # 更新输出
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                flag = (end_counts == min_ends)  # 标记已完成序列
                if flag.any():  # 如果有已完成的
                    for ids in output_ids[flag]:  # 存好已完成序列
                        results.append(ids)
                    flag = (flag == False)  # 标记未完成序列
                    inputs = [i[flag] for i in inputs]  # 只保留未完成部分输入
                    output_ids = output_ids[flag]  # 只保留未完成部分候选集
                    end_counts = end_counts[flag]  # 只保留未完成部分end计数
                    if len(output_ids) == 0:
                        break
        # # 如果还有未完成序列，直接放入结果
        # for ids in output_ids:
        #     results.append(ids)
        # 返回结果
        return results


generator = Generator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)


def gen_sentences(text, n=100, k=20, topp=0.95):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    效果：
        # >>> gen_synonyms(u'微信和支付宝哪个好？')
        [
            u'微信和支付宝，哪个好?',
            u'微信和支付宝哪个好',
            u'支付宝和微信哪个好',
            u'支付宝和微信哪个好啊',
            u'微信和支付宝那个好用？',
            u'微信和支付宝哪个好用',
            u'支付宝和微信那个更好',
            u'支付宝和微信哪个好用',
            u'微信和支付宝用起来哪个好？',
            u'微信和支付宝选哪个好',
        ]
    """
    r = generator.generate(text, n, topp=topp)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort]


def just_show():
    """随机观察一些样本的效果
    """
    some_samples = train_generator.some_samples
    S = [np.random.choice(some_samples) for i in range(3)]
    for s in S:
        try:
            print(u'原句子：%s' % s)
            print(u'同义句子：')
            print(gen_sentences(s, 10, 10))
            print()
        except:
            pass


class Evaluate(keras.callbacks.Callback):
    """评估模型
    """
    def __init__(self, model_dir=None):
        self.lowest = 1e10
        self.latest_path = './latest_model.weights'
        self.best_path = './best_model.weights'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if model_dir is not None:
            self.latest_path = os.path.join(model_dir, 'latest_model.weights')
            self.best_path = os.path.join(model_dir, 'best_model.weights')

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights(self.latest_path)
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(self.best_path)
        # 演示效果
        just_show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.samples", type=str, required=False)
    parser.add_argument("--model_dir", default="saved_model", type=str, required=False)
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--n", default=1, type=int, required=False)
    parser.add_argument("--topp", default=0.8, type=float, required=False)
    parser.add_argument("--text", default="up主喜欢小天使", type=str, required=False)
    parser.add_argument("--prefix", default="传下去，", type=str, required=False)
    args = parser.parse_args()

    if args.do_predict:
        print("input: ", args.text)
        print("="*10, " predict:")
        model.load_weights(os.path.join(args.model_dir, './latest_model.weights'))
        for generation in gen_sentences(args.text, n=args.n, topp=args.topp):
            print(args.prefix + generation)
    else:
        train_generator = data_generator(corpus(args.train), batch_size)
        evaluator = Evaluate(model_dir=args.model_dir)

        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[evaluator]
        )
