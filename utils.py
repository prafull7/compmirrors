def downsample_box_half(x):
    return 0.25 * (
            x[::2, ::2, :] +
            x[1::2, ::2, :] +
            x[1::2, 1::2, :] +
            x[::2, 1::2, :])

def downsample_box_half_5(x):
    return 0.25 * (
            x[:,:,:,::2, ::2] +
            x[:,:,:,1::2, ::2] +
            x[:,:,:,1::2, 1::2] +
            x[:,:,:,::2, 1::2])

def downsample_box_half_4(x):
    return 0.25 * (
            x[:,:,::2, ::2] +
            x[:,:,1::2, ::2] +
            x[:,:,1::2, 1::2] +
            x[:,:,::2, 1::2])


def downsample_box_half_tv(x):
    return 0.25 * (
            x[:,:,::2, ::2,:] +
            x[:,:,1::2, ::2,:] +
            x[:,:,1::2, 1::2,:] +
            x[:,:,::2, 1::2,:])

def downsample_box_half_mono(x):
    return 0.25 * (
            x[::2, ::2] +
            x[1::2, ::2] +
            x[1::2, 1::2] +
            x[::2, 1::2])
