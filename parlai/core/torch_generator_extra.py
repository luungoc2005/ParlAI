from parlai.core.opt import Opt
from parlai.core.torch_agent import History, Batch
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from collections import deque
import torch

class HistoryWithTokenType(History):

    def get_history_vec(self):
        """
        Return a vectorized version of the history.
        """
        if len(self.history_vecs) == 0:
            return None

        if self.vec_type == 'deque':
            history = deque(maxlen=self.max_len)
            for vec in self.history_vecs[:-1]:
                history.extend(vec)
                history.extend(self.delimiter_tok)
            token_type_vec = [1] * len(history)
            history.extend(self.history_vecs[-1])
            if self._global_end_token is not None:
                history.extend([self._global_end_token])
            token_type_vec.extend([2] * (len(history) - len(token_type_vec)))
        else:
            # vec type is a list
            history = []
            for vec in self.history_vecs[:-1]:
                history += vec
                history += self.delimiter_tok
            token_type_vec = [1] * len(history)
            history += self.history_vecs[-1]
            if self._global_end_token is not None:
                history += [self._global_end_token]
            token_type_vec.extend([2] * (len(history) - len(token_type_vec)))
        assert len(history) == len(token_type_vec), f'Size of history vector ({len(history)}) is not equal to size of token type vector ({len(token_type_vec)}) (vec_type: {self.vec_type})'
        return history, token_type_vec

class HistoryWithSepToken(History):

    def __init__(
        self,
        opt,
        field='text',
        vec_type='deque',
        maxlen=None,
        size=-1,
        p1_token='__p1__',
        p2_token='__p2__',
        sep_token='__sep__',
        dict_agent=None,
    ):
        super(HistoryWithSepToken, self).__init__(
            opt=opt,
            field=field,
            vec_type=vec_type,
            maxlen=maxlen,
            size=size,
            p1_token=p1_token,
            p2_token=p2_token,
            dict_agent=dict_agent,
        )
        self.sep_token = sep_token
        self._global_sep_token = self.dict[self.dict.sep_token]

    def get_history_vec(self):
        """
        Return a vectorized version of the history.
        """
        if len(self.history_vecs) == 0:
            return None

        if self.vec_type == 'deque':
            history = deque(maxlen=self.max_len)
            for vec in self.history_vecs[:-1]:
                history.extend(vec)
                history.extend(self.delimiter_tok)
            if self._global_sep_token is not None:
                history.extend([self._global_sep_token])
            history.extend(self.history_vecs[-1])
            if self._global_end_token is not None:
                history.extend([self._global_end_token])
        else:
            # vec type is a list
            history = []
            for vec in self.history_vecs[:-1]:
                history += vec
                history += self.delimiter_tok
            if self._global_sep_token is not None:
                history += [self._global_sep_token]
            history += self.history_vecs[-1]
            if self._global_end_token is not None:
                history += [self._global_end_token]
        return history


class TorchGeneratorWithSepToken(TorchGeneratorAgent):
    SEP_TOKEN = '__sep__'

    @classmethod
    def history_class(cls):
        """
        Return the history class that this agent expects to use.

        Can be overriden if a more complex history is required.
        """
        return HistoryWithSepToken

    def __init__(self, opt: Opt, shared=None):
        super(TorchGeneratorWithSepToken, self).__init__(opt=opt, shared=shared)
        self.SEP_TOKEN = self.dict[self.dict.sep_token]

    def build_history(self):
        """
        Return the constructed history object.
        """
        return self.history_class()(
            self.opt,
            maxlen=self.text_truncate,
            size=self.histsz,
            p1_token=self.P1_TOKEN,
            p2_token=self.P2_TOKEN,
            sep_token=self.SEP_TOKEN,
            dict_agent=self.dict,
        )

class TorchGeneratorWithTokenType(TorchGeneratorAgent):

    @classmethod
    def history_class(cls):
        """
        Return the history class that this agent expects to use.

        Can be overriden if a more complex history is required.
        """
        return HistoryWithTokenType

    def _dummy_batch(self, batchsize, maxlen):
        """
        Create a dummy batch.

        This is used to preinitialize the cuda buffer, or otherwise force a
        null backward pass after an OOM.

        If your model uses additional inputs beyond text_vec and label_vec,
        you will need to override it to add additional fields.
        """
        return Batch(
            text_vec=torch.ones(batchsize, maxlen).long().cuda(),
            tokentype_vec=torch.ones(batchsize, maxlen).long().cuda(),
            label_vec=torch.ones(batchsize, 2).long().cuda(),
            text_lengths=[maxlen] * batchsize,
        )
    
    def _set_text_vec(self, obs, history, truncate):
        """
        Set the 'text_vec' field in the observation.

        Useful to override to change vectorization behavior
        """

        if 'text' not in obs:
            return obs

        if 'text_vec' not in obs:
            # text vec is not precomputed, so we set it using the history
            history_string = history.get_history_str()
            # when text not exist, we get text_vec from history string
            # history could be none if it is an image task and 'text'
            # filed is be empty. We don't want this
            if history_string is None:
                return obs
            obs['full_text'] = history_string
            if history_string:
                obs['text_vec'], obs['tokentype_vec'] = history.get_history_vec()

        # check truncation
        if obs.get('text_vec') is not None:
            truncated_vec = self._check_truncate(obs['text_vec'], truncate, True)
            truncated_tokentype_vec = self._check_truncate(obs['tokentype_vec'], truncate, True)
            obs.force_set('text_vec', torch.LongTensor(truncated_vec))
            obs.force_set('tokentype_vec', torch.LongTensor(truncated_tokentype_vec))
        return obs

    def vectorize(self, *args, **kwargs):
        """
        Override vectorize for generative models.
        """
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = True  # we do want this
        return self.vectorize(*args, **kwargs)

    def vectorize(
        self,
        obs,
        history,
        add_start=True,
        add_end=True,
        text_truncate=None,
        label_truncate=None,
    ):
        """
        Make vectors out of observation fields and store in the observation.

        In particular, the 'text' and 'labels'/'eval_labels' fields are
        processed and a new field is added to the observation with the suffix
        '_vec'.

        If you want to use additional fields on your subclass, you can override
        this function, call super().vectorize(...) to process the text and
        labels, and then process the other fields in your subclass.

        Additionally, if you want to override some of these default parameters,
        then we recommend using a pattern like:

        .. code-block:: python

          def vectorize(self, *args, **kwargs):
              kwargs['add_start'] = False
              return super().vectorize(*args, **kwargs)


        :param obs:
            Single observation from observe function.

        :param add_start:
            default True, adds the start token to each label.

        :param add_end:
            default True, adds the end token to each label.

        :param text_truncate:
            default None, if set truncates text vectors to the specified
            length.

        :param label_truncate:
            default None, if set truncates label vectors to the specified
            length.

        :return:
            the input observation, with 'text_vec', 'label_vec', and
            'cands_vec' fields added.
        """
        self._set_text_vec(obs, history, text_truncate)
        self._set_label_vec(obs, add_start, add_end, label_truncate)
        self._set_label_cands_vec(obs, add_start, add_end, label_truncate)
        return obs

    def batchify(self, obs_batch, sort=False):
        """
        Create a batch of valid observations from an unchecked batch.

        A valid observation is one that passes the lambda provided to the
        function, which defaults to checking if the preprocessed 'text_vec'
        field is present which would have been set by this agent's 'vectorize'
        function.

        Returns a namedtuple Batch. See original definition above for in-depth
        explanation of each field.

        If you want to include additonal fields in the batch, you can subclass
        this function and return your own "Batch" namedtuple: copy the Batch
        namedtuple at the top of this class, and then add whatever additional
        fields that you want to be able to access. You can then call
        super().batchify(...) to set up the original fields and then set up the
        additional fields in your subclass and return that batch instead.

        :param obs_batch:
            List of vectorized observations

        :param sort:
            Default False, orders the observations by length of vectors. Set to
            true when using torch.nn.utils.rnn.pack_padded_sequence.  Uses the text
            vectors if available, otherwise uses the label vectors if available.
        """
        if len(obs_batch) == 0:
            return Batch()

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch()

        valid_inds, exs = zip(*valid_obs)

        # TEXT
        xs, x_lens = None, None
        tok_xs = None
        if any(ex.get('text_vec') is not None for ex in exs):
            _xs = [ex.get('text_vec', self.EMPTY) for ex in exs]
            _tok_xs = [ex.get('tokentype_vec', self.EMPTY) for ex in exs]
            xs, x_lens = self._pad_tensor(_xs)
            tok_xs, _ = self._pad_tensor(_tok_xs)
            if sort:
                sort = False  # now we won't sort on labels
                xs, x_lens, valid_inds, exs = argsort(
                    x_lens, xs, x_lens, valid_inds, exs, descending=True
                )
                tok_xs, _, _, _ = argsort(
                    x_lens, tok_xs, x_lens, valid_inds, exs, descending=True
                )

        # LABELS
        labels_avail = any('labels_vec' in ex for ex in exs)
        some_labels_avail = labels_avail or any('eval_labels_vec' in ex for ex in exs)

        ys, y_lens, labels = None, None, None
        if some_labels_avail:
            field = 'labels' if labels_avail else 'eval_labels'

            label_vecs = [ex.get(field + '_vec', self.EMPTY) for ex in exs]
            labels = [ex.get(field + '_choice') for ex in exs]
            y_lens = [y.shape[0] for y in label_vecs]

            ys, y_lens = self._pad_tensor(label_vecs)

            if sort and xs is None:
                ys, valid_inds, label_vecs, labels, y_lens = argsort(
                    y_lens, ys, valid_inds, label_vecs, labels, y_lens, descending=True
                )

        # LABEL_CANDIDATES
        cands, cand_vecs = None, None
        if any('label_candidates_vecs' in ex for ex in exs):
            cands = [ex.get('label_candidates', None) for ex in exs]
            cand_vecs = [ex.get('label_candidates_vecs', None) for ex in exs]

        # IMAGE
        imgs = None
        if any('image' in ex for ex in exs):
            imgs = [ex.get('image', None) for ex in exs]

        return Batch(
            text_vec=xs,
            text_lengths=x_lens,
            label_vec=ys,
            label_lengths=y_lens,
            labels=labels,
            valid_indices=valid_inds,
            candidates=cands,
            candidate_vecs=cand_vecs,
            image=imgs,
            observations=exs,
            tokentype_vec=tok_xs
        )

    def _model_input(self, batch):
        """
        Create the input (x) value for the model.

        Must return a tuple.  This will be passed directly into the model via
        `*args`, i.e.,

        >>> model(*_model_input(batch))

        This is intentionally overridable so that richer models can pass the
        additional inputs.
        """
        return (batch.text_vec, None, batch.tokentype_vec,)
