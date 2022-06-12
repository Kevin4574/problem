import wrangling_data
import plotly, json
import plotly.graph_objs as go


# read in data
data = wrangling_data.clean_data()
print('===data===')
print(data[0])

country1 = data[0][0]
x1 = data[0][1]
y1 = data[0][2]

# set up chart with plotly, this is setup is very similar with plotly setup in HTML
graph1 = [go.Scatter(x = x1,
                     y = y1,
                     mode = 'lines',
                     name = country1)]
print('===graph===')
print(graph1)

layout = dict(title = 'change in hectares land per person 1990 to 2015',
              xaxis = dict(title = 'year',
                           autotick = False,
                           tick0 = 1990,
                           dtick = 25),
              yaxis = dict(title = 'Hectares')
              )
print('===layout===')
print(layout)

figure = []
figure.append(dict(data = graph1,layout = layout))
print('===figure===')
print(figure)


# convert the plotly figure to Json for java-script and html template
figure_json = json.dumps(figure,
                         cls=plotly.utils.PlotlyJSONEncoder)
print('===json figure===')
print(figure_json)

#create plot ids for html id tag
ids = ['figure-{}'.format(i) for i,_ in enumerate(figure)]
print('===ids===')
print(ids)




