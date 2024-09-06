from controller.RD3G.CarRacing import CarRacing

class DummyMain():

if __name__=="__main__":
    main = CarRacing()
    main.main = DummyMain()
    main.main.track = 

    main.buildDynamicsJacobian()
    main.setup()
    main.solve(save_gif=False,visualize=True,animate=True)
    main.final()
    #main.testAnimation()

