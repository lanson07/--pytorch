import torch
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    # 为了匹配neg是负的，pos是正的
    y_pred = (1 - 2 * y_true) * y_pred
    # 将不属于该类别的设置的很低，做exp后会很小
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    # 构建了一个阈值0，也就是为了匹配原公式中log里的那个1
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    # logsumexp就是LSE，一堆exp求和再取log 
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss
