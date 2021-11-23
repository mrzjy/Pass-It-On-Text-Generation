from bert4keras.models import RoFormer
from keras import Sequential
from keras.layers import Dense
from bert4keras.backend import K
from bert4keras.layers import Embedding, Lambda, Add, LayerNormalization, Dropout


class RoformerLatent(RoFormer):
    def __init__(self, *args, **kwargs):
        super(RoformerLatent, self).__init__(*args, **kwargs)
        self.is_infer = False
        self.latent_size = kwargs.get("latent_size")
        self.inter_size = 150
        # this constraint comes from the fact that we use [usused] bert tokens for latent words (there are 99 of them)
        assert self.latent_size < 100
        self.recognition_network = Dense(self.latent_size, name="Recognition-Network")
        self.bow_network = Sequential(
            layers=[
                Dense(self.inter_size),
                Dense(self.vocab_size, activation="sigmoid")],
            name="BOW-Network"
        )
        self.mode = "response_generation"
        self.special_mask_id = 1
        self.latent_id_start = self.special_mask_id + 1
        self.latent_id_end = self.latent_id_start + self.latent_size

    def call(self, inputs):
        # Embedding (with latent)
        outputs = self.apply_embeddings(inputs)
        # Main
        self.mode = "response_generation"  # affects computing attention bias
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs, i)

        hz = self.apply(
            inputs=outputs,
            layer=Lambda,
            function=lambda x: x[:, self.special_mask_id],
            name='Latent-Representation'
        )
        # Final
        outputs = self.apply_final_layers(outputs)

        # outputs: [cls_representation, mlm_logits, latent_representation]
        outputs.append(self.bow_network(hz))
        return outputs

    def apply_embeddings(self, inputs):
        inputs = inputs[:]
        x = inputs.pop(0)
        s = inputs.pop(0)
        segment_ids = s

        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        if self.shared_segment_embeddings:
            name = 'Embedding-Token'
        else:
            name = 'Embedding-Segment'
        s = self.apply(
            inputs=s,
            layer=Embedding,
            input_dim=2,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            name=name
        )
        x = self.apply(
            inputs=[x, s], layer=Add, name='Embedding-Token-Segment'
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        x = self.apply_latent_embedding(x, segment_ids)
        return x

    def apply_latent_embedding(self, embeddings, segment_ids):
        if self.is_infer:
            return embeddings

        z = self.layer_norm_conds[0]
        if self.shared_segment_embeddings:
            name = 'Embedding-Token'
        else:
            name = 'Embedding-Segment'
        # [batch, 1]
        latent_s = Lambda(lambda a: a[:, self.special_mask_id:self.special_mask_id+1])(segment_ids)
        # [batch, 1, H]
        latent_s = self.apply(
            inputs=latent_s,
            layer=Embedding,
            input_dim=2,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            name=name
        )
        # [batch, 1, H]
        latent_x = self.get_latent_embedding(embeddings)
        latent_x = self.apply(
            inputs=[latent_x, latent_s], layer=Add, name='Embedding-Token-Segment'
        )
        latent_x = self.apply(
            inputs=self.simplify([latent_x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        latent_x = self.apply(
            inputs=latent_x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            latent_x = self.apply(
                inputs=latent_x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        def assign_latent(input_tuple):
            input_embeddings, latent_embedding = input_tuple
            batch = K.shape(input_embeddings)[0]
            length = K.shape(input_embeddings)[1]
            slice1 = K.slice(
                input_embeddings,
                start=[0, 0, 0],
                size=[batch, self.special_mask_id, self.embedding_size])
            slice2 = K.slice(
                input_embeddings,
                start=[0, self.special_mask_id+1, 0],
                size=[batch, length-2, self.embedding_size])
            # concat._keras_shape = (batch, length, self.embedding_size)
            return K.concatenate([slice1, latent_embedding, slice2], axis=1)
        # [batch, L, H]
        x = Lambda(lambda a: assign_latent(a))([embeddings, latent_x])
        return x

    def get_latent_embedding(self, inputs):
        latent_logits = self.call_recognition_network(inputs)
        # [batch, latent_size]
        latent_weights = Lambda(lambda x: gumbel_softmax(x), name="Gumbel-Softmax")(latent_logits)
        # [latent_size, hidden], extract sub embedding matrix for all latent tokens
        latent_embeddings = self.apply(
            inputs=Lambda(lambda x: K.arange(self.latent_id_start, self.latent_id_end))(inputs),
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        # [batch, hidden], compute the pseudo latent embedding
        latent_embedding = Lambda(lambda x: K.dot(x[0], x[1]))([latent_weights, latent_embeddings])
        expand_latent_embedding = Lambda(lambda x: K.expand_dims(x, axis=1))(latent_embedding)
        # [batch, 1, hidden], compute the pseudo latent embedding
        expand_latent_embedding._keras_shape = (None, 1, self.embedding_size)
        return expand_latent_embedding

    def call_recognition_network(self, outputs):
        # go through bert encoder with full self attention
        self.mode = "latent_act_recognition"  # affects computing attention bias
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs, i)
        # get the special mask
        h_M = Lambda(lambda x: x[:, self.special_mask_id], name="get_hM")(outputs)
        # project to latent_size
        return self.recognition_network(h_M)

    def compute_attention_bias(self, inputs=None):
        if self.mode == "response_generation":
            def unilm_mask(s):
                idxs = K.cumsum(s, axis=1)
                mask = idxs[:, None, :] <= idxs[:, :, None]
                mask = K.cast(mask, K.floatx())
                return -(1 - mask[:, None]) * 1e12

            attention_bias = self.apply(
                inputs=self.inputs[1],
                layer=Lambda,
                function=unilm_mask,
                name='Attention-UniLM-Mask'
            )
            return attention_bias
        else:
            return None


def gumbel_softmax(logits, tau=0.67, eps=1e-10):
    """Gumbel softmax."""
    batch_size = K.shape(logits)[0]
    u = K.random_uniform(shape=(batch_size, 1))
    gumbel = 0.0 - K.log(eps - K.log(u + eps))
    y = logits + gumbel
    return K.softmax(y / tau)
