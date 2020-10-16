class Test:
  def __init__(self,):
    pass

  def fun(self,arg):
     print(arg)

t = Test()

callback = t.fun
callback("text")
