from timeit import default_timer as timer

import torch


def time_model(model, input_size):
    model.eval()
    count, duration = 0, 0
    for i in range(50):
        start = timer()
        _ = model(torch.rand(size=input_size))
        if i < 10:
            continue
        duration += timer() - start
        count += 1

    return duration / count


def main():
    from models import Generator
    model = Generator(32, 9)
    duration = time_model(model, [1, 3, 512, 512])
    print("Time Taken (excluding warmup): ", duration)


if __name__ == '__main__':
    main()
