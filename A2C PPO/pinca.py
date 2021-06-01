import pygal as pg
chart = pg.Bar()

chart.add('Fibonacii',[0,1,1,2,3,5,8,13,21])
chart.render_to_file("chart.svg")
