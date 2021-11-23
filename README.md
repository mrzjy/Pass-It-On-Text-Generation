# Pass It On

Corpus + Code + Trained Model for "pass it on" BiliBili meme.

- [Pass It On](#pass-it-on)
    + [Context](#context)
    + [Requirement](#requirement)
    + [Data Description](#data-description)
      - [Explanations](#explanations)
      - [Training Data](#training-data)
    + [Task Description](#task-description)
    + [Model Description](#model-description)
      - [Architecture](#architecture)
      - [Inference](#inference)
        * [Generation Cases](#generation-cases)
      - [Training](#training)

### Context

This repository might help study word-of-mouth rumor propagation, especially on how people reformulate and make a rumor (or a fact) more and more shocking (sometimes funny) when they pass it on and on...

Meanwhile, "pass it on" is also a BiliBili meme. People would write comment on videos or posts trying to deliberately misinterpret what the uploader originally means, resulting in funny word-of-mouth propagation.

### Requirement
~~~
python 3.6
tensorflow 1.14.0
bert4keras 0.10.6
jieba
~~~


### Data Description
We provide a corpus with 3w bilibili comments using the pattern of "pass it on" bilibili meme (namely, comments that start with "pass it on, ")

- Data Distribution

We mainly crawled video/post comments from "top 100 uploaders of the year" for the last 3 years. Several channels with millions of followers such as Genshin Impact are also included as well. Note that the corpus is far from large enough to represent full BiliBili user/comment distribution.

- Data Structure

The original data is saved as "corpus.json" in data folder, and it's a list of dictionary. 
- Two examples (elements) from data/corpus.json 
~~~
{
    "context": {
        "bvid": "BV1Gb4y167gE",
        "uid": "476704454"
    },
    "rumors": [
        {
            "source": "热情的舟山人民把小文哥当海鲜吃了",
            "propagations": []
        },
        {
            "source": "5147779123",
            "propagations": [
                "舟山的海鲜把小文哥吃了",
                "舟山的海鲜想让小文哥吃人",
                "热情的小文哥把海鲜当成舟山人民吃了",
                "小文哥热情地把舟山上的海鲜吃了",
                "热情的海鲜在舟山到处吃小文哥",
                "热情的舟山海鲜把小文哥给吃了。",
                "舟山的热情海鲜把小文哥吃了",
                "小文哥带着热情的海鲜把舟山吃了"
            ]
        },
        {
            "source": "小文哥把舟山人民配海鲜吃了",
            "propagations": []
        }
    ]
},
{
    "context": {
        "bvid": "BV1Bw411d7r8",
        "uid": "476704454"
    },
    "rumors": [
        {
            "source": "小文哥吃了兄弟家一山头的桃",
            "propagations": []
        }
    ]
}
~~~
#### Explanations
- Data Content

All data is collected from BiliBili video or post comments. When someone writes a comment with "pass it on" patterns, others would often follow and leave sub-comments with the same pattern. 
For example,
~~~
a comment : pass it on, the uploader says he likes this girl.
    sub-comment-1: pass it on, the uploader likes to be a girl
    sub-comment-2: pass it on, the uploader likes to be a boy
    sub-comment-3: pass it on, the uploader is a girl
    ...
~~~
For each element in data/corpus.json
~~~
context:   # so that one could refer to source page 
    bvid:  # video (post) id
    uid:   # user (uploader) id
rumors:    # a list containing rumors
    [
        {
            source:  #  source of rumors, might be a comment or just a comment_id (if source has no "pass it on" pattern)
            propagations:  # list of sub-comments, spreading the source in time order
        },
        {source, propagations},
        {source, propagations},
        ...
    ]
~~~

#### Training Data
We also derive "data/train.samples", the training data with common format of "sentence1 \t sentence2" line by line. It contains 290,000 sentence pairs.

### Task Description
We treat the task simply as typical text generation: 
- Input: a rumor
- Output: a spread rumor 

The task is similar to text paraphrasing (generating similar sentences), but this time despite similar sentence formulation, we also want a semantic shift on purpose.

Besides, there's another challenge that lies in the one-to-many relationship between source and propagations.

### Model Description
#### Architecture
We take advantage of [roformer-sim](https://github.com/ZhuiyiTechnology/roformer-sim) pretrained generation-then-rank model as our backbone, since it's a good text paraphrasing baseline.
Besides, in order to model the source-propagations one-to-many relationship as noted before, we add latent variable in a [PLATO](https://arxiv.org/abs/1910.07931) way, where a special token is prepended to input text representing a latent variable. Hence learning is slightly different as well. 

Note: we are not 100% sure that our PLATO-like model implementation is correct \[doge]

We open-source our model for public use. It's a 12-layer modified roformer model.

[Google Drive](https://drive.google.com/drive/folders/1QY7Lhhi_yNFbau0AlC1SkbRVE-h3s0DO)
 
#### Inference
Note $your_model_path the model directory of downloaded model, then you could use command line to perform inference as follows:
~~~
python train_latent.py --do_predict --model_dir=$your_model_path --n=1 --topp=0.8 --text=any_text
~~~
##### Generation Cases
- Example output for "up主喜欢小天使"
~~~
input: up主喜欢小天使
==========  predict:
传下去，up主疑似在天堂被天台感动
传下去，up喜欢天穹
传下去，up主家里有天星
传下去，天使喜欢戴眼镜的up主
传下去，up没有天堂
传下去，up叫天台来烤粉丝。
传下去，up每天早上要直播露脸
传下去，up下天生孩子
传下去，up到天台高产了
传下去，up上天堂了
传下去，up和天使吃了
传下去，楼主的那个帅小伙要穿天堂
传下去，up要cos女巫直播
传下去，天堂大up要爽了
传下去，up下天生孩子翻车了
传下去，up不是人
传下去，天天抽一个up吃8个小孩
传下去，三天之后，up主要被魅魔掏心脏了
~~~
- Example output for "帝君喜欢吃花生米"
~~~
input:  帝君喜欢吃花生米
==========  predict:
传下去，帝君喜欢吃干净的小孩子。
传下去，帝君喜欢捆绑
传下去，帝君用生煎包吃花生米
传下去，帝君吃傻子吃腻了
传下去，喜欢帝君的来自花生米
传下去，花生米吃帝君
传下去，神像吃吃沙子
传下去，我吃帝君屁股
传下去，帝君身边有个米哈游
传下去，帝君只剩干了
传下去，仙跳墙使帝君心疼
传下去，帝君今天上了小通宵
传下去，帝君上床了
传下去，帝君没有下半身
传下去，帝君要炸百京贵妇
传下去，十个视频有了帝君
传下去，帝君会喂食尘神当生日礼物
传下去，视频下一次更新十个帝君
传下去，这个视频里有一年的课代表
~~~
- Example output for "川建国要复辟了"
~~~
input:  川建国要复辟了
==========  predict:
传下去，川建国想要
传下去，川宝上任国君了
传下去，川宝变艾伦了
传下去，《不要传传》
传下去，川宝有天火了。
传下去，阿舅变成了川宝
传下去，川宝长大了也不会忘开
传下去，《川宝要制杖》
传下去，总之，川宝喜欢新衣服
传下去，齐格飞要斩四郎
传下去，老八要吃了川宝
传下去，川普不喜欢制杖
传下去，川团老表是孙笑川
传下去，三叔写盗墓笔记
传下去，川宝没有才浅是制杖
传下去，《川宝喜欢才浅制杖》
传下去，我要吃川宝老爷子
传下去，《我才是川宝喜欢的人》
传下去，全世界辣鸡都不用吃川宝！
传下去，有人冒充川宝想被粉丝上
~~~

#### Training
By default, we train for 10 epochs with batch_size=128. It's encouraged to apply pretrained checkpoint. (e.g., at line 30, checkpoint_path = "chinese_roformer-sim-char-ft_L-12_H-768_A-12")
~~~
python train_latent.py --model_dir=$your_model_dir --train=data/train.samples
~~~