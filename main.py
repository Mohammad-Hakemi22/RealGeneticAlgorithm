from RGA import RGA



if __name__ == "__main__":
    rga = RGA((100, 2),low=[-3, 4.1], high=[12.1, 5.8])
    a, b, c, d = rga.run() # start algorithm
    rga.plot(a, b, c, d)