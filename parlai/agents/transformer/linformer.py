
import torch
import torch.nn as nn
import torch.nn.functional as F


def identity(x, *args, **kwargs):
    return x

def get_act(activation):
    if activation == "gelu":
        return F.gelu
    if activation == "relu":
        return F.relu
    return None

def get_EF(input_size, dim, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    """
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin


class LinearAttention_heads(nn.Module):
    """
    Linear attention, as proposed by the linformer paper
    """
    def __init__(self, dim, dropout, E_proj, F_proj, full_attention=False):
        super(LinearAttention_heads, self).__init__()
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.P_bar = None
        self.full_attention = full_attention

    def forward(self, Q, K, V, **kwargs):
        """
        Assume Q, K, V have same dtype
        E, F are `nn.Linear` modules
        """
        K = K.transpose(1,2)
        if not self.full_attention:
            K = self.E(K)
        Q = torch.matmul(Q, K)

        P_bar = Q/torch.sqrt(torch.tensor(self.dim).type(Q.type()))
        P_bar = P_bar.softmax(dim=-1)

        # Only save this when visualizing
        if "visualize" in kwargs and kwargs["visualize"] == True:
            self.P_bar = P_bar

        P_bar = self.dropout(P_bar)

        if not self.full_attention:
            V = V.transpose(1,2)
            V = self.F(V)
            V = V.transpose(1,2)
        out_tensor = torch.matmul(P_bar, V)

        return out_tensor


class LinformerMultiHeadAttention(nn.Module):
    """
    Multihead attention, with each head being a Linformer Head
    This feeds directly into a feed forward head
    """
    def __init__(self, input_size, dim, channels, dim_k, n_heads, dropout, activation, full_attention, w_o_intermediate_dim=None):
        super(LinformerMultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList()
        self.input_size = input_size
        self.dim_k = dim_k
        self.channels = channels
        self.w_o_intermediate_dim = w_o_intermediate_dim

        E_proj = get_EF(input_size, dim_k)
        F_proj = get_EF(input_size, dim_k)

        self.to_q = nn.ModuleList()
        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()

        for _ in range(n_heads):
            E_proj = get_EF(input_size, dim_k)
            F_proj = get_EF(input_size, dim_k)
            attn = LinearAttention_heads(dim, dropout, E_proj, F_proj, full_attention)
            self.heads.append(attn)
            self.to_q.append(nn.Linear(channels, dim, bias=False))
            self.to_k.append(nn.Linear(channels, dim, bias=False))
            self.to_v.append(nn.Linear(channels, dim, bias=False))
        if w_o_intermediate_dim is None:
            self.w_o = nn.Linear(dim*n_heads, channels)
        else:
            self.w_o_1 = nn.Linear(dim*n_heads, w_o_intermediate_dim)
            self.w_o_2 = nn.Linear(w_o_intermediate_dim, channels)
        self.activation = get_act(activation)

    # def forward(  # type: ignore
    #     # TODO: remove type ignore with pytorch 1.5:
    #     # https://github.com/pytorch/pytorch/pull/31057
    #     self,
    #     query: torch.Tensor,
    #     key: Optional[torch.Tensor] = None,
    #     value: Optional[torch.Tensor] = None,
    #     mask: torch.Tensor = None,
    #     incr_state: Optional[Dict[str, torch.Tensor]] = None,
    #     static_kv: bool = False,
    # ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    def forward(
        self, 
        tensor: torch.Tensor, 
        **kwargs
    ):
        batch_size, input_len, channels = tensor.shape
        head_outputs = []
        for index, head in enumerate(self.heads):
            Q = self.to_q[index](tensor)
            K = self.to_k[index](tensor)
            V = self.to_v[index](tensor)

            head_outputs.append(head(Q,K,V,**kwargs))

        out = torch.cat(head_outputs, dim=-1)
        if self.activation is not None:
            out = self.activation(out)
        if self.w_o_intermediate_dim is None:
            out = self.w_o(out)
        else:
            out = self.w_o_1(out)
            out = self.w_o_2(out)
        return out