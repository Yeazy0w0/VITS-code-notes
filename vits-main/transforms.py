import torch
from torch.nn import functional as F

import numpy as np


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

# 这是一个定义了一个函数 piecewise_rational_quadratic_transform 的代码段。
# 这个函数的作用是计算分段有理二次函数的转换。
# 它的输入有 inputs、unnormalized_widths、unnormalized_heights、unnormalized_derivatives 和 tails。
# 它的输出有 outputs 和 logabsdet。这个函数还有一些可选参数，如 inverse、tail_bound、min_bin_width、min_bin_height 和 min_derivative。
# 其中，inverse 是一个布尔值，表示是否计算反函数。tail_bound 是一个浮点数，表示尾部的边界。min_bin_width、min_bin_height 和 min_derivative 是常量，分别表示最小的区间宽度、高度和导数。
def piecewise_rational_quadratic_transform(inputs, 
                                           unnormalized_widths,
                                           unnormalized_heights,
                                           unnormalized_derivatives,
                                           inverse=False,
                                           tails=None, 
                                           tail_bound=1.,
                                           min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                           min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                           min_derivative=DEFAULT_MIN_DERIVATIVE):

    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {
            'tails': tails,
            'tail_bound': tail_bound
        }

    outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            **spline_kwargs
    )
    return outputs, logabsdet

# searchsorted 函数是一个用于在有序数组中查找某些数的函数。
def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1

# 这是一个定义了一个函数 unconstrained_rational_quadratic_spline 的代码段。
# 这个函数的作用是计算一个无限制的分段有理二次函数的转换。它的输入有 inputs、unnormalized_widths、unnormalized_heights、unnormalized_derivatives 和 tails，它的输出有 outputs 和 logabsdet。
# 这个函数还有一些可选参数，如 inverse、tail_bound、min_bin_width、min_bin_height 和 min_derivative。其中，inverse 是一个布尔值，表示是否计算反函数。tail_bound 是一个浮点数，表示尾部的边界。min_bin_width、min_bin_height 和 min_derivative 是常量，分别表示最小的区间宽度、高度和导数。
# 这个函数首先会调用另一个函数 searchsorted 来计算输入数据的位置，然后判断这些数据是否在一个间隔内，如果不在，则根据 tails 的值进行特殊处理。如果在，则调用 rational_quadratic_spline 函数进行转换。
def unconstrained_rational_quadratic_spline(inputs,
                                            unnormalized_widths,
                                            unnormalized_heights,
                                            unnormalized_derivatives,
                                            inverse=False,
                                            tails='linear',
                                            tail_bound=1.,
                                            min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                            min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                            min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == 'linear':
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    return outputs, logabsdet

# 这是一个定义了一个函数 rational_quadratic_spline 的代码段。
# 这个函数的作用是计算一个分段有理二次函数的转换。
# 它的输入有 inputs、unnormalized_widths、unnormalized_heights、unnormalized_derivatives 和可选参数 left、right、bottom 和 top。
# 它的输出有 outputs 和 logabsdet。这个函数还有一些可选参数，如 inverse、min_bin_width、min_bin_height 和 min_derivative。其中，inverse 是一个布尔值，表示是否计算反函数。min_bin_width、min_bin_height 和 min_derivative 是常量，分别表示最小的区间宽度、高度和导数。
# 这个函数首先会调用另一个函数 searchsorted 来计算输入数据的位置，然后根据这个位置获取对应的宽度、高度、导数和其他参数，最后根据 inverse 的值分两种情况进行计算。
# 如果 inverse 为 False，则计算原函数值；如果 inverse 为 True，则计算反函数值。对于原函数的计算，它会根据有理二次函数的形式求解一个二次方程的根；对于反函数的计算，它会根据有理二次函数的形式求解一个二次方程的未知量。
# 最后，它会计算转换的对数绝对值，这个值表示变换的可逆性。
def rational_quadratic_spline(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=0., right=1., bottom=0., top=1.,
                              min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                              min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                              min_derivative=DEFAULT_MIN_DERIVATIVE):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError('Input to a transform is not within its domain')

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives
                                             + input_derivatives_plus_one
                                             - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives
                                              + input_derivatives_plus_one
                                              - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet
