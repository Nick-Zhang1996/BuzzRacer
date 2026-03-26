''' Empty Car contorller, doesn't do anything, mainly for testing'''
from buzzracer.controllers.controller import Controller, ControllerConfig, ControllerState


@Controller.register(ControllerConfig, ControllerState)
class EmptyController(Controller):
    pass
