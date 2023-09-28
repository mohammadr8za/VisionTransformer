from tensorboard import program
tracking_address = r'D:\mreza\TestProjects\Python\DL\ViT\Experiments\train\BS8_LR1e-05_D0.05_G0.995_L1e-06\runs'

if __name__ == '__main__':
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")